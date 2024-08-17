import unittest

from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from sharedbuffer.SharedBuffer import SharedBuffer
from sharedbuffer.SharedBufferAccessor import SharedBufferAccessor
from util.DeweyNumber import DeweyNumber

import unittest

class TestSharedBufferAccessor(unittest.TestCase):

    def setUp(self):
        config = SharedBufferCacheConfig.from_config('../config.ini')
        self.shared_buffer = SharedBuffer(config)
        self.accessor = SharedBufferAccessor(self.shared_buffer)

    def test_advance_time(self):
        self.accessor.advance_time(1000)
        # Check if the time has advanced properly in shared buffer
        # Add your assertions here

    def test_register_event(self):
        event_id = self.accessor.register_event("event1", 1000)
        self.assertEqual(event_id.id, 0)
        self.assertEqual(event_id.timestamp, 1000)

        lockable_event = self.shared_buffer.get_event(event_id)
        self.assertIsNotNone(lockable_event)
        self.assertEqual(lockable_event.element, "event1")
        self.assertEqual(lockable_event.ref_counter, 1)

    def test_put(self):
        event_id = self.accessor.register_event("event1", 1000)
        version = DeweyNumber([1])
        state_name = "state1"
        current_node_id = self.accessor.put(state_name, event_id, None, version)

        lockable_node = self.shared_buffer.get_entry(current_node_id)
        self.assertIsNotNone(lockable_node)
        self.assertEqual(lockable_node.element.get_edges()[0].get_target(), None)
        self.assertEqual(lockable_node.element.get_edges()[0].get_dewey_number(), version)

    def test_extract_patterns(self):
        event_id = self.accessor.register_event("event1", 1000)
        version = DeweyNumber([1])
        state_name = "state1"
        current_node_id = self.accessor.put(state_name, event_id, None, version)

        patterns = self.accessor.extract_patterns(current_node_id, version)
        self.assertIsNotNone(patterns)
        self.assertGreater(len(patterns), 0)

    def test_materialize_match(self):
        event_id = self.accessor.register_event("event1", 1000)
        version = DeweyNumber([1])
        state_name = "state1"
        current_node_id = self.accessor.put(state_name, event_id, None, version)

        patterns = self.accessor.extract_patterns(current_node_id, version)
        match = patterns[0]
        materialized_match = self.accessor
        self.assertIn(state_name, materialized_match)
        self.assertEqual(materialized_match[state_name][0], "event1")

    def test_lock_and_release_node(self):
        event_id = self.accessor.register_event("event1", 1000)
        version = DeweyNumber([1])
        state_name = "state1"
        current_node_id = self.accessor.put(state_name, event_id, None, version)

        self.accessor.lock_node(current_node_id, version)
        lockable_node = self.shared_buffer.get_entry(current_node_id)
        self.assertEqual(lockable_node.ref_counter, 1)

        self.accessor.release_node(current_node_id, version)
        lockable_node = self.shared_buffer.get_entry(current_node_id)
        self.assertIsNone(lockable_node)

    def test_lock_and_release_event(self):
        event_id = self.accessor.register_event("event1", 1000)

        self.accessor.lock_event(event_id)
        lockable_event = self.shared_buffer.get_event(event_id)
        self.assertEqual(lockable_event.ref_counter, 2)

        self.accessor.release_event(event_id)
        lockable_event = self.shared_buffer.get_event(event_id)
        self.assertEqual(lockable_event.ref_counter, 1)

    def tearDown(self):
        self.accessor.close()


if __name__ == '__main__':
    unittest.main()