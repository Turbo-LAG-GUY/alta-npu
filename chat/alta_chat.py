#!/usr/bin/env python3
"""Alta Chat (2026-11-24): a chat window for the 8-bit LLM engine on the A311D -- the prompt runs on the NPU,
the reply is generated on the CPU from the NPU's own weight buffers. CPU mode = everything on the CPU.
Run: chat/alta-chat (the launcher preloads the NPU guard: nothing in this window can open the NPU's node)."""
import os
import sys
import threading
import time

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E  # noqa: E402

APP_ID = "io.github.turbolagguy.AltaChat"
CSS = b"""
.bubble { padding: 10px 14px; border-radius: 16px; }
.bubble.user { background: alpha(@accent_bg_color, 0.9); color: @accent_fg_color; }
.bubble.bot { background: alpha(@card_bg_color, 1); border: 1px solid alpha(@borders, 0.6); }
.bubble.note { background: transparent; color: alpha(@window_fg_color, 0.6); font-size: smaller; }
.stats { font-size: smaller; color: alpha(@window_fg_color, 0.65); font-feature-settings: "tnum"; }
.composer { padding: 8px 10px; border-radius: 14px; background: @card_bg_color; border: 1px solid alpha(@borders, .7); }
"""
BACKENDS = [("npu", "NPU prompt + CPU reply"), ("cpu", "CPU only")]


class Bubble(Gtk.Box):
    def __init__(self, kind, text=""):
        super().__init__(orientation=Gtk.Orientation.VERTICAL)
        self.label = Gtk.Label(label=text, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR, xalign=0, selectable=True)
        self.label.set_max_width_chars(70)
        self.label.add_css_class("bubble")
        self.label.add_css_class(kind)
        self.append(self.label)
        self.set_halign(Gtk.Align.END if kind == "user" else Gtk.Align.START)
        self.set_margin_start(60 if kind == "user" else 12)
        self.set_margin_end(12 if kind == "user" else 60)
        self.set_margin_top(4)
        self.set_margin_bottom(4)
        self.meta = None
        if kind == "bot":
            self.meta = Gtk.Label(xalign=0, css_classes=["stats"])
            self.meta.set_margin_start(6)
            self.meta.set_margin_top(2)
            self.append(self.meta)

    def set_text(self, t):
        self.label.set_text(t)


class Window(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Alta Chat", default_width=760, default_height=860)
        self.client = None
        self.conv = None
        self.model = None
        self.backend = None
        self.busy = False
        self.stop_flag = False
        self.settings = dict(temp=0.7, top_k=40, top_p=0.9, rep_pen=1.1, max_new=384)

        tv = Adw.ToolbarView()
        hb = Adw.HeaderBar()
        self.title = Adw.WindowTitle(title="Alta Chat", subtitle="no model loaded")
        hb.set_title_widget(self.title)
        self.new_btn = Gtk.Button(icon_name="list-add-symbolic", tooltip_text="New chat")
        self.new_btn.connect("clicked", lambda *_: self.new_chat())
        hb.pack_start(self.new_btn)
        menu = Gio.Menu()
        menu.append("Unload model", "win.unload")
        menu.append("Engine log", "win.log")
        menu.append("About Alta Chat", "win.about")
        hb.pack_end(Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu, tooltip_text="Menu"))
        self.set_btn = Gtk.MenuButton(icon_name="preferences-system-symbolic", tooltip_text="Generation settings",
                                      popover=self.settings_popover())
        hb.pack_end(self.set_btn)
        for name, cb in (("unload", self.unload), ("log", self.show_log), ("about", self.about)):
            a = Gio.SimpleAction.new(name, None)
            a.connect("activate", lambda *_a, f=cb: f())
            self.add_action(a)
        tv.add_top_bar(hb)

        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.stack.add_named(self.setup_page(), "setup")
        self.stack.add_named(self.chat_page(), "chat")
        tv.set_content(self.stack)
        self.toasts = Adw.ToastOverlay(child=tv)
        self.set_content(self.toasts)
        self.connect("close-request", self.on_close)
        self.new_btn.set_sensitive(False)
        GLib.timeout_add(500, self.poll)
        if E.unit_active():                         # an engine from an earlier window: reattach
            self.load_status.set_text("Reconnecting to the running engine...")
            self.loading = True
        else:
            self.loading = False

    # ---------- pages
    def setup_page(self):
        sp = Adw.StatusPage(icon_name=APP_ID, title="Alta Chat",
                            description="A small language model on the A311D's NPU.\nThe prompt runs on the NPU, "
                                        "the reply is written on the CPU from the NPU's own weight buffers.")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14, halign=Gtk.Align.CENTER, width_request=380)
        grp = Adw.PreferencesGroup()
        self.model_row = Adw.ComboRow(title="Model", model=Gtk.StringList.new(list(E.MODELS)))
        self.model_row.set_subtitle("135M loads in seconds; 1.7B takes ~1.5 min and ~2.3 GB")
        self.back_row = Adw.ComboRow(title="Runs on", model=Gtk.StringList.new([b[1] for b in BACKENDS]))
        grp.add(self.model_row)
        grp.add(self.back_row)
        box.append(grp)
        self.load_btn = Gtk.Button(label="Load model", css_classes=["suggested-action", "pill"],
                                   halign=Gtk.Align.CENTER)
        self.load_btn.connect("clicked", lambda *_: self.load())
        box.append(self.load_btn)
        self.progress = Gtk.ProgressBar(visible=False)
        box.append(self.progress)
        self.load_status = Gtk.Label(wrap=True, justify=Gtk.Justification.CENTER, css_classes=["stats"])
        box.append(self.load_status)
        sp.set_child(box)
        return sp

    def chat_page(self):
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.scroll = Gtk.ScrolledWindow(vexpand=True, hscrollbar_policy=Gtk.PolicyType.NEVER)
        self.msgs = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.msgs.set_margin_top(10)
        self.msgs.set_margin_bottom(10)
        clamp = Adw.Clamp(maximum_size=820, child=self.msgs)
        self.scroll.set_child(clamp)
        outer.append(self.scroll)
        comp = Gtk.Box(spacing=8, css_classes=["composer"])
        comp.set_margin_start(12)
        comp.set_margin_end(12)
        comp.set_margin_bottom(6)
        self.entry = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, hexpand=True, accepts_tab=False)
        self.entry.set_size_request(-1, 44)
        kc = Gtk.EventControllerKey()
        kc.connect("key-pressed", self.on_key)
        self.entry.add_controller(kc)
        es = Gtk.ScrolledWindow(child=self.entry, hexpand=True, max_content_height=160, propagate_natural_height=True,
                                hscrollbar_policy=Gtk.PolicyType.NEVER)
        comp.append(es)
        self.send_btn = Gtk.Button(icon_name="go-up-symbolic", css_classes=["suggested-action", "circular"],
                                   valign=Gtk.Align.END, tooltip_text="Send (Enter)")
        self.send_btn.connect("clicked", lambda *_: self.send_or_stop())
        comp.append(self.send_btn)
        outer.append(Adw.Clamp(maximum_size=820, child=comp))
        self.stats = Gtk.Label(xalign=0.5, css_classes=["stats"])
        self.stats.set_margin_bottom(8)
        outer.append(self.stats)
        return outer

    def settings_popover(self):
        pop = Gtk.Popover()
        g = Gtk.Grid(row_spacing=8, column_spacing=12)
        for m in ("set_margin_start", "set_margin_end", "set_margin_top", "set_margin_bottom"):
            getattr(g, m)(12)
        specs = [("Temperature (0 = greedy)", "temp", 0, 2, 0.05, 2), ("Top-k", "top_k", 0, 200, 1, 0),
                 ("Top-p", "top_p", 0.05, 1, 0.05, 2), ("Repetition penalty", "rep_pen", 1, 2, 0.05, 2),
                 ("Max reply tokens", "max_new", 16, 1024, 16, 0)]
        for i, (lab, key, lo, hi, st, dg) in enumerate(specs):
            g.attach(Gtk.Label(label=lab, xalign=0), 0, i, 1, 1)
            sb = Gtk.SpinButton.new_with_range(lo, hi, st)
            sb.set_digits(dg)
            sb.set_value(self.settings[key])
            sb.connect("value-changed", lambda w, k=key, d=dg: self.settings.__setitem__(k, w.get_value() if d
                                                                                            else int(w.get_value())))
            g.attach(sb, 1, i, 1, 1)
        g.attach(Gtk.Label(label="System prompt", xalign=0), 0, len(specs), 2, 1)
        self.sys_entry = Gtk.Entry(text=E.SYSTEM, width_chars=40)
        self.sys_entry.set_tooltip_text("Takes effect with the next new chat")
        g.attach(self.sys_entry, 0, len(specs) + 1, 2, 1)
        pop.set_child(g)
        return pop

    # ---------- engine lifecycle
    def load(self):
        self.model = list(E.MODELS)[self.model_row.get_selected()]
        self.backend = BACKENDS[self.back_row.get_selected()][0]
        bad = E.preflight(self.backend)
        if bad:
            self.load_status.set_text("\n".join(bad))
            return
        warn = E.warnings(self.model)
        if warn and not getattr(self, "warned", False):
            self.warned = True                        # shown once; pressing Load again goes ahead
            self.load_status.set_text("\n".join(warn) + "\n\nPress Load model again to go ahead.")
            return
        self.warned = False
        r = E.start(self.model, self.backend)
        if r.returncode:
            self.load_status.set_text("could not start the engine: " + (r.stderr or r.stdout).strip())
            return
        self.loading = True
        self.t_load = time.monotonic()
        self.load_btn.set_sensitive(False)
        self.progress.set_visible(True)
        self.progress.set_fraction(0)
        self.load_status.set_text("Starting the engine...")

    def poll(self):
        if self.loading:
            st = E.status()
            w = st.split()
            if w and w[0] == "loading":
                l, n, ms, avail = int(w[1]), int(w[2]), float(w[3]), w[4]
                self.progress.set_fraction(l / max(n, 1))
                self.load_status.set_text("Loading and compiling layer %d of %d for the NPU  (%.0f s, %s MB free)"
                                          % (l, n, ms / 1e3, avail) if self.backend != "cpu" else
                                          "Loading layer %d of %d  (%.0f s)" % (l, n, ms / 1e3))
            elif w and w[0] == "ready":
                self.connect_engine()
            elif w and w[0] == "error":
                self.load_failed(st)
            elif not E.unit_active() and getattr(self, "t_load", 0) and time.monotonic() - self.t_load > 3:
                self.load_failed("the engine stopped while loading")
            elif not E.unit_active() and not getattr(self, "t_load", 0):
                self.loading = False
                self.load_status.set_text("")
        elif self.client and not self.busy and not E.unit_active():
            self.client = None
            self.go_setup("The engine has stopped.\n" + E.journal_tail(4))
        return True

    def load_failed(self, why):
        self.loading = False
        self.load_btn.set_sensitive(True)
        self.progress.set_visible(False)
        self.load_status.set_text("%s\n\n%s" % (why, E.journal_tail(8)))

    def connect_engine(self):
        try:
            self.client = E.Client()
        except OSError:
            return                                   # socket not accepting yet: next poll
        self.loading = False
        info = self.client.info
        if not self.model:                           # reattached: find the model from the packs path
            self.model = next((k for k, m in E.MODELS.items() if m["packs"] == info["packs"]), list(E.MODELS)[0])
            self.backend = "cpu" if info["backend"].startswith("cpu") else "npu"
        where = "NPU + CPU" if self.backend == "npu" else "CPU"
        self.title.set_subtitle("%s · %s · %d-token context" % (self.model, where, info["ctx"]))
        self.progress.set_visible(False)
        self.load_btn.set_sensitive(True)
        self.new_btn.set_sensitive(True)
        self.new_chat()
        self.stack.set_visible_child_name("chat")
        self.entry.grab_focus()

    def go_setup(self, msg=""):
        self.stack.set_visible_child_name("setup")
        self.title.set_subtitle("no model loaded")
        self.new_btn.set_sensitive(False)
        self.load_btn.set_sensitive(True)
        self.progress.set_visible(False)
        self.load_status.set_text(msg)

    def unload(self):
        if self.busy:
            self.toast("Stop the reply first")
            return
        if self.client:
            self.client.quit()
            self.client = None
        elif E.unit_active():
            subprocess_stop()
        self.go_setup("Model unloaded.")

    # ---------- chat
    def new_chat(self):
        if self.busy:
            return
        c = self.msgs.get_first_child()
        while c:
            n = c.get_next_sibling()
            self.msgs.remove(c)
            c = n
        self.conv = E.Conversation(E.MODELS[self.model]["tok"], self.sys_entry.get_text().strip() or E.SYSTEM)
        self.add_bubble("note", "New chat with %s. Enter sends, Shift+Enter adds a line." % self.model)
        self.stats.set_text("")

    def add_bubble(self, kind, text=""):
        b = Bubble(kind, text)
        self.msgs.append(b)
        GLib.idle_add(self.scroll_end)
        return b

    def scroll_end(self):
        adj = self.scroll.get_vadjustment()
        adj.set_value(adj.get_upper())
        return False

    def on_key(self, ctl, keyval, code, state):
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and not state & Gdk.ModifierType.SHIFT_MASK:
            self.send_or_stop()
            return True
        return False

    def send_or_stop(self):
        if self.busy:
            self.stop_flag = True
            try:
                self.client.stop()
            except OSError:
                pass
            return
        buf = self.entry.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        if not text or not self.client:
            return
        buf.set_text("")
        self.add_bubble("user", text)
        self.conv.add_user(text)
        bot = self.add_bubble("bot", "…")
        self.busy = True
        self.stop_flag = False
        self.send_btn.set_icon_name("media-playback-stop-symbolic")
        self.send_btn.set_tooltip_text("Stop")
        threading.Thread(target=self.worker, args=(list(self.conv.ids), bot), daemon=True).start()

    def worker(self, ids, bot):
        toks, info, err = [], {}, None
        t0 = time.monotonic()
        try:
            s = self.settings
            for ev in self.client.gen(ids, int(s["max_new"]), s["temp"], int(s["top_k"]), s["top_p"], s["rep_pen"],
                                      int(time.time() * 1000) & 0x7FFFFFFF):
                if ev[0] == "TOK":
                    toks.append(ev[1])
                    if len(toks) == 1:
                        info["first"] = time.monotonic() - t0
                    GLib.idle_add(self.update_bot, bot, self.conv.text(toks))
                elif ev[0] == "PREFILL":
                    info["prefill"] = ev
                    GLib.idle_add(self.stats.set_text, "prompt: %d new tokens in %.2f s (%s) · writing..." %
                                  (ev[3], ev[1] / 1e3, ev[2]))
                elif ev[0] == "END":
                    info["end"] = ev
        except Exception as e:  # noqa: BLE001
            err = str(e)
        GLib.idle_add(self.finish, bot, toks, info, err)

    def update_bot(self, bot, text):
        bot.set_text(text or "…")
        self.scroll_end()
        return False

    def finish(self, bot, toks, info, err):
        self.busy = False
        self.send_btn.set_icon_name("go-up-symbolic")
        self.send_btn.set_tooltip_text("Send (Enter)")
        if err:
            bot.set_text(self.conv.text(toks) or "(no reply)")
            self.toast("Engine: " + err)
            if "context full" in err:
                self.add_bubble("note", "The conversation fills the context. Start a new chat (+).")
            self.conv.add_reply(toks)
            return False
        self.conv.add_reply(toks)
        bot.set_text(self.conv.text(toks) or "(empty reply)")
        p, e = info.get("prefill"), info.get("end")
        if e:
            why = {"stop": "", "length": " · hit the length limit", "user": " · stopped",
                   "context": " · context full"}[e[1]]
            tps = 1e3 / e[3] if e[3] > 0 else 0
            bot.meta.set_text("%d tokens · %.1f tok/s%s" % (e[2], tps, why))
            path = {"npu-chunk": "NPU", "cpu-chunk": "CPU chunk", "decode": "CPU"}.get(p[2], p[2]) if p else "?"
            self.stats.set_text("prompt: %d new tokens in %.2f s on the %s · reply: %.1f tok/s · context %d / %d"
                                % (p[3], p[1] / 1e3, path, tps, len(self.conv.ids), self.client.info["ctx"]))
        return False

    # ---------- misc
    def toast(self, t):
        self.toasts.add_toast(Adw.Toast(title=t, timeout=4))

    def show_log(self):
        d = Adw.MessageDialog(transient_for=self, heading="Engine log", body=E.journal_tail(25) or "(empty)")
        d.add_response("ok", "Close")
        d.present()

    def about(self):
        Adw.AboutWindow(transient_for=self, application_name="Alta Chat", application_icon=APP_ID,
                        version="0.1 (2026-11-24)", developer_name="Jordan",
                        comments="SmolLM2 on the Amlogic A311D: 8-bit prompt processing on the NPU (etnaviv), "
                                 "replies on the CPU from the NPU's weight buffers.",
                        website="https://github.com/Turbo-LAG-GUY/alta-npu").present()

    def on_close(self, *_):
        if not E.unit_active():
            return False
        d = Adw.MessageDialog(transient_for=self, heading="Unload the model?",
                              body="Keeping it loaded makes the next start instant, but it holds memory "
                                   "(and, in NPU mode, the NPU) until you unload it.")
        d.add_response("keep", "Keep loaded")
        d.add_response("unload", "Unload")
        d.set_response_appearance("unload", Adw.ResponseAppearance.SUGGESTED)
        d.set_default_response("unload")

        def resp(_d, r):
            if r == "unload":
                if self.busy and self.client:
                    self.client.stop()
                if self.client:
                    self.client.quit()
                else:
                    subprocess_stop()
            elif self.client:
                self.client.s.close()
            self.get_application().quit()
        d.connect("response", resp)
        d.present()
        return True


def subprocess_stop():
    """no connection: ask the engine to quit over a fresh one, else stop the unit"""
    try:
        E.Client().quit()
    except Exception:  # noqa: BLE001
        import subprocess
        subprocess.run(["systemctl", "--user", "stop", E.UNIT], capture_output=True)


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)

    def do_activate(self):
        w = self.props.active_window
        if not w:
            prov = Gtk.CssProvider()
            prov.load_from_data(CSS)
            Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), prov,
                                                      Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
            icons = Gtk.IconTheme.get_for_display(Gdk.Display.get_default())
            icons.add_search_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons"))
            w = Window(self)
        w.present()


if __name__ == "__main__":
    sys.exit(App().run(sys.argv))
