#pragma once

#include <unordered_map>
#include <list>
#include <iostream>
#include <mutex>

template<typename K, typename V>
class LRUCache {
private:
    // Capacity of cache
    int m_capacity;
    std::mutex m;
    // List to store key-value pairs in order of use
    // Most recently used at front, least recently used at back
    std::list<std::pair<K, V>> m_items;
    
    // Hash map to store key to iterator mappings
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> m_item_map;

public:
    explicit LRUCache(int capacity) : m_capacity(capacity) {
        std::cout << "LRUCache created with capacity: " << capacity << std::endl;
    }
    
    // Get value by key. Returns nullptr if not found
    V* get(const K& key) {
        std::lock_guard<std::mutex> guard(m);
        auto it = m_item_map.find(key);
        if (it == m_item_map.end()) {
            return nullptr;
        }
        
        // Move accessed item to front (most recently used)
        m_items.splice(m_items.begin(), m_items, it->second);
        return &(it->second->second);
    }
    
    // Put key-value pair into cache
    void put(const K& key, const V& value) {
        std::lock_guard<std::mutex> guard(m);
        auto it = m_item_map.find(key);
        
        if (it != m_item_map.end()) {
            // Key exists, update value and move to front
            it->second->second = value;
            m_items.splice(m_items.begin(), m_items, it->second);
            return;
        }
        
        // Check if cache is full
        if (m_capacity >= 0 && int(m_items.size()) >= m_capacity) {
            // Remove least recently used item (back of list)
            auto last = m_items.back();
            m_item_map.erase(last.first);
            m_items.pop_back();
        }
        
        // Insert new item at front
        m_items.emplace_front(key, value);
        m_item_map[key] = m_items.begin();
    }
    
    // Clear the cache
    void clear() {
        std::lock_guard<std::mutex> guard(m);
        m_items.clear();
        m_item_map.clear();
    }
    
    // Get current size
    size_t size() const {
        return m_items.size();
    }
    
    // Get capacity
    int capacity() const {
        return m_capacity;
    }
};