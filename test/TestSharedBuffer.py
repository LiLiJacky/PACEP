import unittest
from datetime import timedelta

from configuration.SharedBufferCacheConfig import SharedBufferCacheConfig
from sharedbuffer.EventId import EventId
from sharedbuffer.Lockable import Lockable
from sharedbuffer.NodeId import NodeId
from sharedbuffer.ShareBufferNode import SharedBufferNode
from sharedbuffer.SharedBuffer import SharedBuffer


class TestSharedBuffer(unittest.TestCase):

    def setUp(self):
        self.config = SharedBufferCacheConfig.from_config('../config.ini')
        self.buffer = SharedBuffer(self.config)

    def test_register_event(self):
        event_id = self.buffer.register_event("event1", 1000)
        self.assertEqual(event_id.id, 0)
        self.assertEqual(event_id.timestamp, 1000)

        lockable_event = self.buffer.get_event(event_id)
        self.assertIsNotNone(lockable_event)
        self.assertEqual(lockable_event.element, "event1")
        self.assertEqual(lockable_event.ref_counter, 1)

    def test_upsert_event(self):
        event_id = EventId(1, 1000)
        lockable_event = Lockable("event1", 1)
        self.buffer.upsert_event(event_id, lockable_event)

        cached_event = self.buffer.get_event(event_id)
        self.assertIsNotNone(cached_event)
        self.assertEqual(cached_event.element, "event1")
        self.assertEqual(cached_event.ref_counter, 1)

    def test_remove_event(self):
        event_id = self.buffer.register_event("event1", 1000)
        self.buffer.remove_event(event_id)

        cached_event = self.buffer.get_event(event_id)
        self.assertIsNone(cached_event)

    def test_upsert_entry(self):
        node_id = NodeId("node1", "example1")
        shared_buffer_node = SharedBufferNode()
        lockable_node = Lockable(shared_buffer_node, 1)
        self.buffer.upsert_entry(node_id, lockable_node)

        cached_entry = self.buffer.get_entry(node_id)
        self.assertIsNotNone(cached_entry)
        self.assertEqual(cached_entry.element, shared_buffer_node)
        self.assertEqual(cached_entry.ref_counter, 1)

    def test_remove_entry(self):
        node_id = NodeId("node1", "example1")
        shared_buffer_node = SharedBufferNode()
        lockable_node = Lockable(shared_buffer_node, 1)
        self.buffer.upsert_entry(node_id, lockable_node)
        self.buffer.remove_entry(node_id)

        cached_entry = self.buffer.get_entry(node_id)
        self.assertIsNone(cached_entry)

    def test_is_empty(self):
        self.assertTrue(self.buffer.is_empty())
        self.buffer.register_event("event1", 1000)
        self.assertFalse(self.buffer.is_empty())

    def test_advance_time(self):
        self.buffer.register_event("event1", 1000)
        self.buffer.register_event("event2", 2000)
        self.buffer.advance_time(1500)

        self.assertNotIn(1000, self.buffer.events_count)
        self.assertIn(2000, self.buffer.events_count)

if __name__ == '__main__':
    unittest.main()