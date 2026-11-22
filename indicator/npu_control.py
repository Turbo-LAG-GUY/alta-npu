#!/usr/bin/python3
"""npu-control: the NPU job-queue window for the alta etnaviv driver (A311D NPU).

Opened from the NPU status indicator's menu ("Job queue…"), or by itself:

    npu_control.py            the settings window
    npu_control.py --trip     the same, opened because the queue dropped to 1 after a hang

What it changes (all under /sys/module/etnaviv/parameters, see npu_queue.py):
  - how many jobs may wait in the NPU's own queue at once (1-4),
  - whether a recovered hang drops that to 1 (and what to do when it did:
    put it back, keep 1 for now, or reboot),
  - whether programs take turns or share the queue,
  - program priority by process name (high / normal / low), saved and applied
    again at every login by the indicator.

It never opens the NPU device or sends it work. Writing the settings needs the
udev rule from alta-npu root step 250; without it the window is read-only.
"""

import argparse
import os
import sys

import npu_queue as q

APP_ID = "org.altanpu.NpuControl"
RESP_RESTORE, RESP_KEEP, RESP_REBOOT = 1, 2, 3

LIMIT_HINT = ("More than 1 lets the next job wait inside the NPU, so it doesn't sit idle "
              "between jobs. Jobs still run one after another.")
AUTO_HINT = "If it happens, you're asked whether to put it back or reboot."
TURNS_HINT = "Off: programs share the queue, like a graphics driver."
GUARD_HINT = ("After a higher-priority job, lower-priority jobs may only enter an empty queue "
              "for this long, so they can't fill it just before the next one. 0 turns it off.")
PRIORITY_HELP = ("Priority is matched by process name (as in ps or top, up to 15 characters) when "
                 "a program opens the NPU; programs already running move at their next idle moment. "
                 "Higher priority goes first, the others run in the gaps. Every Python script is "
                 "named python3 (or python), so they share one setting.")


class ControlPanel:
    """The window's content, a Gtk.Box in self.widget (also used off-screen by --snapshot)."""

    def __init__(self, Gtk, window=None, trip=False):
        self.Gtk = Gtk
        self.window = window
        self.trip_requested = trip
        self.st = q.QueueState()
        self.cfg = q.load_config()
        self._updating = False
        self._rows = None
        self._trip = None

        V, H = Gtk.Orientation.VERTICAL, Gtk.Orientation.HORIZONTAL
        self.widget = Gtk.Box(orientation=V, spacing=12)
        for side in ("top", "bottom", "start", "end"):
            getattr(self.widget, f"set_margin_{side}")(16)

        # Banners -----------------------------------------------------------
        self.bar_driver, self.lbl_driver = self._bar(Gtk.MessageType.WARNING, "dialog-information")
        self.bar_trip, self.lbl_trip = self._bar(Gtk.MessageType.ERROR, "dialog-warning")
        self.btn_restore = self.bar_trip.add_button("Put back", RESP_RESTORE)
        self.bar_trip.add_button("Keep 1 for now", RESP_KEEP)
        self.bar_trip.add_button("Reboot…", RESP_REBOOT)
        self.bar_trip.connect("response", self._on_trip_response)
        self.bar_msg, self.lbl_msg = self._bar(Gtk.MessageType.INFO)
        self.bar_msg.set_show_close_button(True)
        self.bar_msg.connect("response", lambda bar, _r: self._hide_bar(bar))

        # Job queue ---------------------------------------------------------
        grid = Gtk.Grid(column_spacing=24, row_spacing=14)
        for side in ("top", "bottom", "start", "end"):
            getattr(grid, f"set_margin_{side}")(12)
        self.queue_frame = self._frame("Job queue", grid)

        self.spin_limit = Gtk.SpinButton.new_with_range(q.LIMIT_MIN, q.LIMIT_MAX, 1)
        self.spin_limit.connect("value-changed", self._on_limit)
        self._row(grid, 0, "Jobs waiting in the NPU at once", self.spin_limit, LIMIT_HINT)

        self.sw_auto = Gtk.Switch()
        self.sw_auto.connect("notify::active", self._on_auto)
        self._row(grid, 1, "Drop to 1 after a recovered hang", self.sw_auto, AUTO_HINT)

        self.sw_turns = Gtk.Switch()
        self.sw_turns.connect("notify::active", self._on_turns)
        self._row(grid, 2, "Programs take turns", self.sw_turns, TURNS_HINT)

        guard = Gtk.Box(orientation=H, spacing=6)
        self.spin_guard = Gtk.SpinButton.new_with_range(0, q.GUARD_MAX_MS, 10)
        self.spin_guard.connect("value-changed", self._on_guard)
        guard.pack_start(self.spin_guard, False, False, 0)
        guard.pack_start(Gtk.Label(label="ms"), False, False, 0)
        self._row(grid, 3, "Priority guard", guard, GUARD_HINT)

        self.lbl_status = Gtk.Label(xalign=0, wrap=True)
        grid.attach(self.lbl_status, 0, 4, 2, 1)

        self.chk_login = Gtk.CheckButton(label="Apply these settings and priorities when I log in")
        self.chk_login.connect("toggled", self._on_login)
        grid.attach(self.chk_login, 0, 5, 2, 1)

        # Program priority --------------------------------------------------
        box = Gtk.Box(orientation=V, spacing=8)
        for side in ("top", "bottom", "start", "end"):
            getattr(box, f"set_margin_{side}")(12)
        self.prio_frame = self._frame("Program priority", box)

        # name, level, status, saved
        self.store = Gtk.ListStore(str, str, str, bool)
        self.view = Gtk.TreeView(model=self.store, headers_visible=True)
        self.view.get_selection().connect("changed", lambda *_: self._update_buttons())

        col = Gtk.TreeViewColumn("Program", Gtk.CellRendererText(), text=0)
        col.set_expand(True)
        self.view.append_column(col)

        levels = Gtk.ListStore(str)
        for level in q.LEVELS:
            levels.append([level])
        combo = Gtk.CellRendererCombo(model=levels, text_column=0, has_entry=False, editable=True)
        combo.connect("edited", self._on_level_edited)
        self.view.append_column(Gtk.TreeViewColumn("Priority", combo, text=1))
        self.view.append_column(Gtk.TreeViewColumn("Now", Gtk.CellRendererText(), text=2))

        scroll = Gtk.ScrolledWindow(min_content_height=150, propagate_natural_height=True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_shadow_type(Gtk.ShadowType.IN)
        scroll.add(self.view)
        box.pack_start(scroll, True, True, 0)

        buttons = Gtk.Box(orientation=H, spacing=6)
        self.btn_add = Gtk.Button(label="Add program…")
        self.btn_add.connect("clicked", self._on_add)
        self.btn_remove = Gtk.Button(label="Remove")
        self.btn_remove.connect("clicked", self._on_remove)
        buttons.pack_start(self.btn_add, False, False, 0)
        buttons.pack_start(self.btn_remove, False, False, 0)
        box.pack_start(buttons, False, False, 0)
        box.pack_start(self._hint(PRIORITY_HELP), False, False, 0)

        self.widget.pack_start(self.queue_frame, False, False, 0)
        self.widget.pack_start(self.prio_frame, True, True, 0)

    # -- building blocks ---------------------------------------------------

    def _bar(self, kind, icon=None):
        Gtk = self.Gtk
        bar = Gtk.InfoBar()
        bar.set_message_type(kind)
        area = bar.get_content_area()
        if icon:
            image = Gtk.Image.new_from_icon_name(icon, Gtk.IconSize.DIALOG)
            area.pack_start(image, False, False, 0)
            image.show()
        label = Gtk.Label(xalign=0, wrap=True)
        label.set_max_width_chars(60)
        area.pack_start(label, True, True, 0)
        bar.set_no_show_all(True)
        label.show()
        self.widget.pack_start(bar, False, False, 0)
        return bar, label

    def _show_bar(self, bar, label, text):
        if label.get_text() != text:
            label.set_text(text)
        if not bar.get_visible():
            bar.show()
            bar.set_revealed(True)

    def _hide_bar(self, bar):
        if bar.get_visible():
            bar.set_revealed(False)
            bar.hide()

    def _frame(self, title, child):
        Gtk = self.Gtk
        frame = Gtk.Frame()
        label = Gtk.Label(xalign=0)
        label.set_markup(f"<b>{title}</b>")
        frame.set_label_widget(label)
        frame.add(child)
        return frame

    def _hint(self, text):
        label = self.Gtk.Label(label=text, xalign=0, wrap=True)
        label.set_max_width_chars(60)
        label.get_style_context().add_class("dim-label")
        return label

    def _row(self, grid, row, title, control, hint):
        """Title with its hint underneath on the left, the control on the right."""
        Gtk = self.Gtk
        left = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, hexpand=True)
        left.pack_start(Gtk.Label(label=title, xalign=0), False, False, 0)
        if hint:
            h = self._hint(hint)
            h.get_style_context().add_class("caption")
            left.pack_start(h, False, False, 0)
        grid.attach(left, 0, row, 1, 1)
        control.set_valign(Gtk.Align.CENTER)
        control.set_halign(Gtk.Align.END)
        grid.attach(control, 1, row, 1, 1)

    def message(self, text):
        self._show_bar(self.bar_msg, self.lbl_msg, text)

    # -- reading -------------------------------------------------------------

    def tick(self):
        try:
            self.refresh()
        except Exception as e:  # one bad read must not close the window
            self.message(f"Could not read the driver's settings: {e}")
        return True

    def refresh(self):
        st = self.st = q.read_state()
        self.cfg = q.load_config()
        self._updating = True
        try:
            self._refresh_banners(st)
            self._refresh_queue(st)
            self._refresh_programs(st)
        finally:
            self._updating = False

    def _refresh_banners(self, st):
        if not st.loaded:
            text = "The NPU driver (etnaviv) is not loaded."
        elif not st.runtime:
            text = (f"The loaded driver fixes the queue at boot ({st.limit} at once) and has no "
                    f"program priority. Root step 250 installs the alta driver with patches 0006 "
                    f"and 0007, which this window needs.")
        elif not st.writable:
            text = ("Read-only: your user may not change the driver's settings. Root step 250 "
                    "gives it write access.")
        elif not st.priority:
            text = "This driver has no program priority (patch 0007), so that part is read-only."
        else:
            text = None
        if text:
            self._show_bar(self.bar_driver, self.lbl_driver, text)
        else:
            self._hide_bar(self.bar_driver)

        self._trip = q.unanswered_trip(st) if st.runtime else None
        if self._trip:
            frm, n = self._trip
            times = "once" if n == 1 else f"{n} times"
            text = (f"The NPU hung and the driver recovered it ({times} this boot), so the queue "
                    f"dropped from {frm} to 1 job at a time.")
            self._show_bar(self.bar_trip, self.lbl_trip, text)
            self.btn_restore.set_label(f"Put back to {frm}")
            self.btn_restore.set_visible(frm > 1)
            self.bar_trip.set_response_sensitive(RESP_RESTORE, st.writable and frm > 1)
            self.bar_trip.set_response_sensitive(RESP_KEEP, True)
        else:
            self._hide_bar(self.bar_trip)

    def _refresh_queue(self, st):
        can = st.runtime and st.writable
        for w in (self.spin_limit, self.sw_auto, self.sw_turns, self.chk_login):
            w.set_sensitive(can)
        self.spin_guard.set_sensitive(can and st.guard_ms is not None)
        if st.limit is not None and self.spin_limit.get_value_as_int() != st.limit:
            self.spin_limit.set_value(st.limit)
        if st.auto is not None and self.sw_auto.get_active() != st.auto:
            self.sw_auto.set_active(st.auto)
        if st.turns is not None and self.sw_turns.get_active() != st.turns:
            self.sw_turns.set_active(st.turns)
        if st.guard_ms is not None and self.spin_guard.get_value_as_int() != st.guard_ms:
            self.spin_guard.set_value(st.guard_ms)
        if self.chk_login.get_active() != bool(self.cfg["apply_at_login"]):
            self.chk_login.set_active(bool(self.cfg["apply_at_login"]))

        if not st.loaded:
            status = "driver not loaded"
        elif not st.runtime:
            status = f"stock or older alta driver: {st.limit} at once, set at boot"
        else:
            trips = "no trips this boot" if not st.trips else (
                f"dropped to 1 after a hang: {st.trips}× this boot (last from {st.trip_from})")
            saved = self.cfg["limit"]
            login = f" · at login: {saved}" if saved is not None else ""
            status = f"{trips}{login}"
        self.lbl_status.set_text("Status: " + status)

    def _program_rows(self, st):
        running = {}
        for c in st.clients:
            running.setdefault(c.name, []).append(c)
        names = sorted(set(self.cfg["priorities"]) | set(running), key=str.lower)
        rows = []
        for name in names:
            saved = name in self.cfg["priorities"]
            clients = running.get(name, [])
            level = self.cfg["priorities"][name] if saved else (clients[0].priority if clients else "normal")
            if clients:
                pids = ", ".join(str(c.pid) for c in clients)
                now = f"pid {pids}: {clients[0].priority}"
                if not all(c.applied for c in clients):
                    now += ", moves when idle"
            else:
                now = "not running"
            rows.append((name, level, now, saved))
        return rows

    def _refresh_programs(self, st):
        self.prio_frame.set_sensitive(st.priority)
        rows = self._program_rows(st)
        if rows != self._rows:
            selected = self._selected_name()
            self.store.clear()
            for row in rows:
                it = self.store.append(list(row))
                if row[0] == selected:
                    self.view.get_selection().select_iter(it)
            self._rows = rows
        self._update_buttons()

    def _selected_name(self):
        model, it = self.view.get_selection().get_selected()
        return model[it][0] if it else None

    def _update_buttons(self):
        can = self.st.priority and self.st.writable
        self.btn_add.set_sensitive(can)
        model, it = self.view.get_selection().get_selected()
        self.btn_remove.set_sensitive(bool(can and it and model[it][3]))

    # -- writing -------------------------------------------------------------

    def _write(self, what, fn, *args):
        try:
            fn(*args)
            return True
        except (q.NoAccess, ValueError, OSError) as e:
            self.message(f"Could not change {what}: {e}")
            self.refresh()
            return False

    def _save(self):
        try:
            q.save_config(self.cfg)
        except OSError as e:
            self.message(f"Could not save the settings: {e}")

    def _on_limit(self, spin):
        if self._updating:
            return
        n = spin.get_value_as_int()
        if self._write("the queue", q.set_limit, n):
            self.cfg["limit"] = n
            self._save()

    def _on_auto(self, sw, _pspec):
        if self._updating:
            return
        if self._write("the automatic drop", q.set_auto, sw.get_active()):
            self.cfg["auto"] = sw.get_active()
            self._save()

    def _on_turns(self, sw, _pspec):
        if self._updating:
            return
        if self._write("turn taking", q.set_turns, sw.get_active()):
            self.cfg["turns"] = sw.get_active()
            self._save()

    def _on_guard(self, spin):
        if self._updating:
            return
        ms = spin.get_value_as_int()
        if self._write("the priority guard", q.set_guard, ms):
            self.cfg["guard_ms"] = ms
            self._save()

    def _on_login(self, chk):
        if self._updating:
            return
        self.cfg["apply_at_login"] = chk.get_active()
        self._save()

    def _set_priorities(self, priorities):
        named = [n for n, l in priorities.items() if l != "normal"]
        if len(named) > q.MAX_NAMES:
            self.message(f"The driver holds at most {q.MAX_NAMES} programs with a high or low "
                         f"priority; set one back to normal first.")
            return False
        if not self._write("program priority", q.set_priorities, priorities):
            return False
        self.cfg["priorities"] = priorities
        self._save()
        self.refresh()
        return True

    def _on_level_edited(self, _renderer, path, level):
        if level not in q.LEVELS:
            return
        name = self.store[path][0]
        priorities = dict(self.cfg["priorities"])
        priorities[name] = level
        self._set_priorities(priorities)

    def _on_remove(self, _button):
        name = self._selected_name()
        if name and name in self.cfg["priorities"]:
            priorities = dict(self.cfg["priorities"])
            del priorities[name]
            self._set_priorities(priorities)

    def _on_add(self, _button):
        Gtk = self.Gtk
        dlg = Gtk.Dialog(title="Add program", transient_for=self.window, modal=True)
        dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Add", Gtk.ResponseType.OK)
        dlg.set_default_response(Gtk.ResponseType.OK)
        area = dlg.get_content_area()
        area.set_spacing(8)
        for side in ("top", "bottom", "start", "end"):
            getattr(area, f"set_margin_{side}")(12)
        area.pack_start(Gtk.Label(label="Process name, as in ps or top (up to 15 characters):",
                                  xalign=0), False, False, 0)
        entry = Gtk.Entry(activates_default=True, max_length=q.NAME_MAX)
        area.pack_start(entry, False, False, 0)
        level = Gtk.ComboBoxText()
        for lv in q.LEVELS:
            level.append(lv, lv)
        level.set_active_id("high")
        area.pack_start(level, False, False, 0)
        err = Gtk.Label(xalign=0)
        area.pack_start(err, False, False, 0)
        dlg.show_all()
        while dlg.run() == Gtk.ResponseType.OK:
            name = entry.get_text()
            problem = q.check_name(name)
            if problem:
                err.set_text(f"Can't use that name: {problem}.")
                continue
            priorities = dict(self.cfg["priorities"])
            priorities[name] = level.get_active_id()
            if self._set_priorities(priorities):
                break
        dlg.destroy()

    def _on_trip_response(self, _bar, response):
        st, trip = self.st, self._trip
        if not trip:
            return
        frm, _n = trip
        if response == RESP_RESTORE:
            if self._write("the queue", q.set_limit, frm):
                q.answer_trip(st)
                self.message(f"The queue is back to {frm} jobs at once.")
        elif response == RESP_KEEP:
            q.answer_trip(st)
            self.message("Keeping 1 job at a time until you change it here or reboot.")
        elif response == RESP_REBOOT:
            if self._confirm_reboot():
                q.answer_trip(st)
                q.reboot()
                return
        self.refresh()

    def _confirm_reboot(self):
        Gtk = self.Gtk
        dlg = Gtk.MessageDialog(transient_for=self.window, modal=True,
                                message_type=Gtk.MessageType.QUESTION,
                                text="Reboot now?")
        dlg.format_secondary_text("Save your work in other programs first. "
                                  "The NPU starts clean after the reboot.")
        dlg.add_buttons("Cancel", Gtk.ResponseType.CANCEL, "Reboot", Gtk.ResponseType.OK)
        ok = dlg.run() == Gtk.ResponseType.OK
        dlg.destroy()
        return ok


# --------------------------------------------------------------------------


def _gtk():
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import GLib, Gtk
    except (ImportError, ValueError) as e:
        sys.exit(f"npu-control: needs PyGObject with GTK 3 ({e}). Run it with /usr/bin/python3.")
    if not Gtk.init_check(sys.argv[:1])[0]:
        sys.exit("npu-control: no graphical session found.")
    return Gtk, GLib


def run_window(trip, unique=True):
    Gtk, GLib = _gtk()
    from gi.repository import Gio
    flags = Gio.ApplicationFlags.FLAGS_NONE if unique else Gio.ApplicationFlags.NON_UNIQUE
    app = Gtk.Application(application_id=APP_ID, flags=flags)

    def activate(app):
        for win in app.get_windows():
            win.present()
            return
        win = Gtk.ApplicationWindow(application=app, title="NPU job queue")
        win.set_default_size(720, 640)
        win.set_icon_name("utilities-system-monitor")
        panel = ControlPanel(Gtk, win, trip=trip)
        win.add(panel.widget)
        win.show_all()
        panel.tick()
        GLib.timeout_add(1000, panel.tick)

    app.connect("activate", activate)
    return app.run(sys.argv[:1])


def snapshot(path, width):
    """Draw the window off-screen into a PNG (for tests; shows nothing on the desktop)."""
    Gtk, _GLib = _gtk()
    win = Gtk.OffscreenWindow()
    panel = ControlPanel(Gtk, None)
    panel.widget.set_size_request(width, -1)
    win.add(panel.widget)
    win.show_all()
    panel.refresh()
    for _ in range(50):
        while Gtk.events_pending():
            Gtk.main_iteration()
    win.get_pixbuf().savev(path, "png", [], [])
    print(path)


def main():
    parser = argparse.ArgumentParser(description="NPU job-queue settings (alta etnaviv driver).")
    parser.add_argument("--trip", action="store_true",
                        help="opened because the queue dropped to 1 after a hang")
    parser.add_argument("--root", default="/", help=argparse.SUPPRESS)       # fake /sys + /proc
    parser.add_argument("--snapshot", metavar="PNG", help=argparse.SUPPRESS)  # off-screen test
    parser.add_argument("--width", type=int, default=720, help=argparse.SUPPRESS)
    args = parser.parse_args()
    q.ROOT = os.path.abspath(args.root)
    if args.snapshot:
        snapshot(args.snapshot, args.width)
        return 0
    return run_window(args.trip, unique=(q.ROOT == "/"))


if __name__ == "__main__":
    sys.exit(main())
