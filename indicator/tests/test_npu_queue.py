"""Tests for npu_queue against a fake /sys + /proc tree (no NPU, no root).

    /usr/bin/python3 -m unittest discover -s tests      (from the npu-status folder)
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import npu_queue as q  # noqa: E402

PARAMS = {
    "hw_job_limit": "1",
    "hw_job_auto": "Y",
    "hw_job_trips": "0",
    "hw_job_trip_from": "0",
    "hw_job_turns": "N",
    "hw_job_prio_guard_ms": "100",
    "hw_job_prio": "",
    "hw_job_clients": "",
    "keep_powered": "Y",
}


def make_tree(root, params=PARAMS, boot="boot-a"):
    pdir = os.path.join(root, "sys/module/etnaviv/parameters")
    os.makedirs(pdir, exist_ok=True)
    for name, value in params.items():
        with open(os.path.join(pdir, name), "w") as f:
            f.write(value + "\n")
    os.makedirs(os.path.join(root, "proc/sys/kernel/random"), exist_ok=True)
    with open(os.path.join(root, "proc/sys/kernel/random/boot_id"), "w") as f:
        f.write(boot + "\n")


def param(root, name, value=None):
    path = os.path.join(root, "sys/module/etnaviv/parameters", name)
    if value is None:
        with open(path) as f:
            return f.read().strip()
    with open(path, "w") as f:
        f.write(value + "\n")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "root")
        self.home = os.path.join(self.tmp.name, "config")
        make_tree(self.root)
        self._env = os.environ.get("XDG_CONFIG_HOME")
        os.environ["XDG_CONFIG_HOME"] = self.home
        self._root, self._stop = q.ROOT, q.STOP_FILE
        q.ROOT = self.root
        q.STOP_FILE = "/home/test/alta-npu/runs/STOP"

    def tearDown(self):
        q.ROOT, q.STOP_FILE = self._root, self._stop
        if self._env is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = self._env
        self.tmp.cleanup()


class Parsing(unittest.TestCase):
    def test_table_round_trip(self):
        t = q.parse_table("hl2_linux=high, jb_engine-f85c5=low,bad,x=urgent,=high\n")
        self.assertEqual(t, {"hl2_linux": "high", "jb_engine-f85c5": "low"})
        self.assertEqual(q.format_table({"b": "low", "a": "high", "c": "normal"}), "a=high,b=low")

    def test_table_cap(self):
        many = {f"p{i:02d}": "high" for i in range(20)}
        self.assertEqual(len(q.format_table(many).split(",")), q.MAX_NAMES)

    def test_clients(self):
        cs = q.parse_clients("4251 hl2_linux high pending\n77 Web Content normal applied\njunk\n")
        self.assertEqual([(c.pid, c.name, c.priority, c.applied) for c in cs],
                         [(4251, "hl2_linux", "high", False), (77, "Web Content", "normal", True)])

    def test_names(self):
        self.assertIsNone(q.check_name("jb_engine-f85c5"))
        self.assertIsNotNone(q.check_name("jb_engine-f85c58fa"))   # 18 > 15
        self.assertIsNotNone(q.check_name("a,b"))
        self.assertIsNotNone(q.check_name(" x"))
        self.assertIsNotNone(q.check_name(""))


class Reading(Base):
    def test_full_driver(self):
        param(self.root, "hw_job_prio", "hl2_linux=high")
        param(self.root, "hw_job_clients", "4251 hl2_linux high applied")
        st = q.read_state()
        self.assertTrue(st.loaded and st.runtime and st.priority and st.writable)
        self.assertEqual((st.limit, st.auto, st.turns, st.guard_ms), (1, True, False, 100))
        self.assertEqual(st.table, {"hl2_linux": "high"})
        self.assertEqual(st.clients[0].name, "hl2_linux")
        self.assertIn("up to 1 at once", q.summary(st))

    def test_stock_driver(self):
        tmp = tempfile.mkdtemp(dir=self.tmp.name)
        make_tree(tmp, {"hw_job_limit": "4", "job_hang_limit": "0"})
        q.ROOT = tmp
        st = q.read_state()
        self.assertTrue(st.loaded)
        self.assertFalse(st.runtime or st.priority or st.writable)
        self.assertEqual(q.summary(st), "Job queue: 4 at once (fixed at boot)")
        self.assertEqual(q.apply_saved(st, dict(q.DEFAULT_CONFIG, limit=4)), [])

    def test_not_loaded(self):
        q.ROOT = os.path.join(self.tmp.name, "empty")
        st = q.read_state()
        self.assertFalse(st.loaded)
        self.assertIsNone(q.summary(st))


class Writing(Base):
    def test_setters(self):
        q.set_limit(4)
        q.set_auto(False)
        q.set_turns(True)
        q.set_guard(250)
        q.set_priorities({"hl2_linux": "high", "python3": "normal", "jb": "low"})
        self.assertEqual([param(self.root, n) for n in
                          ("hw_job_limit", "hw_job_auto", "hw_job_turns", "hw_job_prio_guard_ms",
                           "hw_job_prio")],
                         ["4", "N", "Y", "250", "hl2_linux=high,jb=low"])
        with self.assertRaises(ValueError):
            q.set_limit(5)
        with self.assertRaises(ValueError):
            q.write_param("keep_powered", "N")

    def test_no_access(self):
        if os.geteuid() == 0:
            self.skipTest("root can write read-only files")
        os.chmod(os.path.join(self.root, "sys/module/etnaviv/parameters/hw_job_limit"), 0o444)
        self.assertFalse(q.read_state().writable)
        with self.assertRaises(q.NoAccess):
            q.set_limit(2)


class Login(Base):
    def cfg(self, **kw):
        cfg = dict(q.DEFAULT_CONFIG, **kw)
        q.save_config(cfg)
        return cfg

    def test_applies_once_per_boot(self):
        self.cfg(limit=4, turns=True, priorities={"hl2_linux": "high"})
        self.assertEqual(q.apply_at_login(), [])
        self.assertEqual(param(self.root, "hw_job_limit"), "4")
        self.assertEqual(param(self.root, "hw_job_turns"), "Y")
        self.assertEqual(param(self.root, "hw_job_prio"), "hl2_linux=high")
        param(self.root, "hw_job_limit", "2")          # the user changes it by hand
        q.apply_at_login()                              # indicator restarted, same boot
        self.assertEqual(param(self.root, "hw_job_limit"), "2")
        make_tree(self.root, dict(PARAMS), boot="boot-b")  # next boot
        q.apply_at_login()
        self.assertEqual(param(self.root, "hw_job_limit"), "4")

    def test_off(self):
        self.cfg(limit=4, apply_at_login=False)
        q.apply_at_login()
        self.assertEqual(param(self.root, "hw_job_limit"), "1")

    def test_keeps_1_after_trip(self):
        param(self.root, "hw_job_trips", "1")
        param(self.root, "hw_job_trip_from", "4")
        self.cfg(limit=4)
        msgs = q.apply_at_login()
        self.assertEqual(param(self.root, "hw_job_limit"), "1")
        self.assertTrue(msgs and "after a hang" in msgs[0])

    def test_keeps_1_with_stop_latch(self):
        stop = os.path.join(self.root, "home/test/alta-npu/runs/STOP")
        os.makedirs(os.path.dirname(stop))
        open(stop, "w").close()
        self.cfg(limit=4, priorities={"hl2_linux": "high"})
        msgs = q.apply_at_login()
        self.assertEqual(param(self.root, "hw_job_limit"), "1")
        self.assertEqual(param(self.root, "hw_job_prio"), "hl2_linux=high")  # priorities still apply
        self.assertIn("runs/STOP", msgs[0])

    def test_no_access_message(self):
        if os.geteuid() == 0:
            self.skipTest("root can write read-only files")
        os.chmod(os.path.join(self.root, "sys/module/etnaviv/parameters/hw_job_limit"), 0o444)
        self.cfg(limit=4)
        msgs = q.apply_at_login()
        self.assertIn("root step 250", msgs[0])


class Trips(Base):
    def test_watch_and_answer(self):
        w = q.TripWatch()
        self.assertIsNone(w.check(q.read_state()))
        param(self.root, "hw_job_trips", "1")
        param(self.root, "hw_job_trip_from", "4")
        param(self.root, "hw_job_limit", "1")
        st = q.read_state()
        self.assertEqual(w.check(st), (4, 1))
        self.assertIsNone(w.check(st))                  # reported once
        self.assertEqual(q.TripWatch().check(st), (4, 1))  # indicator restart: still unanswered
        q.answer_trip(st)
        self.assertIsNone(q.TripWatch().check(st))      # answered
        self.assertIsNone(q.unanswered_trip(st))
        param(self.root, "hw_job_trips", "2")
        self.assertEqual(q.unanswered_trip(q.read_state()), (4, 2))  # a second trip asks again
        with open(q.state_file()) as f:
            self.assertEqual(json.load(f)["trip_answered"], {"boot": "boot-a", "trips": 1})


if __name__ == "__main__":
    unittest.main()
