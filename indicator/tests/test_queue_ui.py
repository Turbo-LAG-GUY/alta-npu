"""The indicator's job-queue hooks and the window's handlers, on a fake /sys tree.

Nothing is shown on the desktop: the indicator part uses stand-in menu items, and the
window part builds its content off-screen (skipped without a graphical session).
"""

import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import npu_queue as q  # noqa: E402
import npu_status as ns  # noqa: E402
from test_npu_queue import Base, param  # noqa: E402


class FakeItem:
    def __init__(self):
        self.label, self.visible = "", False

    def get_label(self):
        return self.label

    def set_label(self, text):
        self.label = text

    def get_visible(self):
        return self.visible

    def show(self):
        self.visible = True

    def hide(self):
        self.visible = False


class Indicator(Base):
    def setUp(self):
        super().setUp()
        self.sent, self.opened = [], []
        self._notify = ns.send_notification
        ns.send_notification = lambda *a, **k: self.sent.append(a)
        self.app = ns.TrayApp.__new__(ns.TrayApp)   # no Gtk, no tray icon
        self.app.queue = FakeItem()
        self.app.trip_watch = q.TripWatch()
        self.app._open_queue_window = lambda trip=False: self.opened.append(trip)

    def tearDown(self):
        ns.send_notification = self._notify
        super().tearDown()

    def test_menu_row(self):
        self.app._check_queue()
        self.assertTrue(self.app.queue.visible)
        self.assertEqual(self.app.queue.label, "Job queue: up to 1 at once")

    def test_trip_warns_once_and_opens_window(self):
        param(self.root, "hw_job_trips", "1")
        param(self.root, "hw_job_trip_from", "4")
        self.app._check_queue()
        self.app._check_queue()
        self.assertEqual(len(self.sent), 1)
        self.assertIn("from 4 to 1", self.sent[0][1])
        self.assertEqual(self.opened, [True])
        self.assertIn("dropped to 1 after a hang", self.app.queue.label)

    def test_login_message(self):
        q.save_config(dict(q.DEFAULT_CONFIG, limit=4))
        param(self.root, "hw_job_trips", "1")
        self.app._queue_at_login()
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(param(self.root, "hw_job_limit"), "1")


def gtk_or_skip():
    try:
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk
    except (ImportError, ValueError):
        raise unittest.SkipTest("no PyGObject")
    if not Gtk.init_check([])[0]:
        raise unittest.SkipTest("no graphical session")
    return Gtk


class Window(Base):
    def setUp(self):
        super().setUp()
        Gtk = gtk_or_skip()
        import npu_control
        self.win = Gtk.OffscreenWindow()
        self.panel = npu_control.ControlPanel(Gtk, None)
        self.win.add(self.panel.widget)
        self.win.show_all()
        self.panel.refresh()

    def tearDown(self):
        self.win.destroy()
        super().tearDown()

    def test_limit_and_switches(self):
        self.panel.spin_limit.set_value(3)
        self.panel.sw_turns.set_active(True)
        self.panel.spin_guard.set_value(200)
        self.assertEqual(param(self.root, "hw_job_limit"), "3")
        self.assertEqual(param(self.root, "hw_job_turns"), "Y")
        self.assertEqual(param(self.root, "hw_job_prio_guard_ms"), "200")
        cfg = q.load_config()
        self.assertEqual((cfg["limit"], cfg["turns"], cfg["guard_ms"]), (3, True, 200))

    def test_refresh_does_not_write(self):
        param(self.root, "hw_job_limit", "2")     # changed outside the window
        self.panel.refresh()
        self.assertEqual(self.panel.spin_limit.get_value_as_int(), 2)
        self.assertIsNone(q.load_config()["limit"])   # reading it back saved nothing

    def test_priority_by_name(self):
        param(self.root, "hw_job_clients", "4251 hl2_linux normal applied\n")
        self.panel.refresh()
        self.assertEqual(self.panel.store[0][0], "hl2_linux")
        self.panel._on_level_edited(None, "0", "high")
        self.assertEqual(param(self.root, "hw_job_prio"), "hl2_linux=high")
        self.assertEqual(q.load_config()["priorities"], {"hl2_linux": "high"})
        self.panel.view.get_selection().select_path(0)
        self.panel._on_remove(None)
        self.assertEqual(param(self.root, "hw_job_prio"), "")
        self.assertEqual(q.load_config()["priorities"], {})

    def test_trip_put_back_and_keep(self):
        param(self.root, "hw_job_trips", "1")
        param(self.root, "hw_job_trip_from", "4")
        self.panel.refresh()
        self.assertTrue(self.panel.bar_trip.get_visible())
        self.assertEqual(self.panel.btn_restore.get_label(), "Put back to 4")
        self.panel._on_trip_response(None, 1)       # RESP_RESTORE
        self.assertEqual(param(self.root, "hw_job_limit"), "4")
        self.assertFalse(self.panel.bar_trip.get_visible())
        param(self.root, "hw_job_trips", "2")
        param(self.root, "hw_job_limit", "1")
        self.panel.refresh()
        self.assertTrue(self.panel.bar_trip.get_visible())
        self.panel._on_trip_response(None, 2)       # RESP_KEEP
        self.assertEqual(param(self.root, "hw_job_limit"), "1")
        self.assertFalse(self.panel.bar_trip.get_visible())

    def test_read_only_driver(self):
        os.chmod(os.path.join(self.root, "sys/module/etnaviv/parameters/hw_job_limit"), 0o444)
        self.panel.refresh()
        self.assertTrue(self.panel.bar_driver.get_visible())
        self.assertFalse(self.panel.spin_limit.get_sensitive())


if __name__ == "__main__":
    unittest.main()
