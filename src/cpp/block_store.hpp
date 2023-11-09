/*
 * \file block_store.hpp
 *
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_BLOCK_STORE_HPP
#define VERITAS_BLOCK_STORE_HPP

#include <vector>
#include <iostream>

namespace veritas {

class BlockStoreOOM : public std::exception {
public:
    virtual const char* what() const noexcept override {
        return "BlockStore: out of memory";
    }
};


/**
 * Store immutable dynamically-sized arrays of type T in stable memory.
 * Pointers returned by `BlockStore::store` are stable for as long as this
 * object lives.
 */
template <typename T>
class BlockStore {
public:
    using Block = std::vector<T>;
    using const_iterator = typename Block::const_iterator;

private:
    std::vector<Block> blocks_;

    Block& get_block_with_remaining_capacity(size_t cap, size_t rem_memory_capacity) {
        Block& block = blocks_.back();
        if (block.capacity() - block.size() < cap) {
            size_t rem_capacity = rem_memory_capacity / sizeof(T);
            if (rem_capacity > block.capacity() * 2) // double size of blocks each time,..
                rem_capacity = block.capacity() * 2; // .. unless memory limit almost reached
            else if (rem_capacity > 0)
                std::cerr << "WARNING: almost running out of memory, "
                    << static_cast<double>(rem_memory_capacity) / (1024.0*1024.0)
                    << " mb left " << std::endl;
            else
                throw BlockStoreOOM();

            Block new_block;
            new_block.reserve(rem_capacity);
            blocks_.push_back(std::move(new_block));

            //std::cout << "BlockStore memory: " << get_mem_size() << " bytes, "
            //    << (static_cast<double>(get_mem_size()) / (1024.0 * 1024.0)) << " mb ("
            //    << blocks_.size() << " blocks)" << std::endl;
        }

        return blocks_.back();
    }


public:
    struct Ref {
        const_iterator begin;
        const_iterator end;
    };

    const static size_t MIN_BLOCK_SIZE = 5 * 1024 * 1024; // 5MB

    /**
     * @param min_block_size size in bytes of the first block in the store
     */
    BlockStore(size_t min_block_size = MIN_BLOCK_SIZE) : blocks_{} {
        Block block;
        block.reserve(min_block_size / sizeof(T));
        blocks_.push_back(std::move(block));
    }

    // disallow: references change wrt other blockstore, most likely this is a mistake
    BlockStore(const BlockStore&) = delete;
    BlockStore& operator=(const BlockStore&) = delete;
    BlockStore(BlockStore&& o) { std::swap(blocks_, o.blocks_); }
    BlockStore& operator=(BlockStore&& o) {
        std::swap(blocks_, o.blocks_); return *this;
    }

    size_t get_mem_size() const {
        size_t mem = 0;
        for (const Block& b : blocks_)
            mem += b.capacity() * sizeof(T);
        return mem;
    }

    size_t get_used_mem_size() const {
        size_t mem = 0;
        for (const Block& b : blocks_)
            mem += b.size() * sizeof(T);
        return mem;
    }

    template <typename IT>
    Ref store(IT begin, IT end, size_t rem_memory_capacity) {
        // this store_ block has enough space to accomodate the workspace
        size_t size = end - begin;
        Block& block = get_block_with_remaining_capacity(size, rem_memory_capacity);

        // push a copy of the workspace
        size_t start_index = block.size();
        for (; begin != end; ++begin)
            block.push_back(*begin);

        auto b = block.begin() + start_index;
        auto e = b + size;

        return { b, e };
    }

    template <typename Container>
    Ref store(const Container& c, size_t rem_memory_capacity)
    { return store(c.begin(), c.end(), rem_memory_capacity); }
};

} // namespace veritas

#endif // VERITAS_BLOCK_STORE_HPP
