#include "lru_cache.h"
#include <iostream>

// Example usage
int main() {
    // Create cache with capacity 3
    LRUCache<int, std::string> cache(-1);
    
    // Add some items
    cache.put(1, "one");
    cache.put(2, "two");
    cache.put(3, "three");
    
    // Get an item (moves it to front)
    if (auto value = cache.get(2)) {
        std::cout << "Found value: " << *value << std::endl;
    }
    
    // Add new item, causing least recently used to be evicted
    cache.put(4, "four");
    
    // Try to get evicted item
    if (cache.get(1) == nullptr) {
        std::cout << "Item 1 was evicted" << std::endl;
    }
    
    // Print current size
    std::cout << "Cache size: " << cache.size() << std::endl;
    
    return 0;
}