/*
 * Copyright 2022 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_BLOCK_STORE_HPP
#define VERITAS_BLOCK_STORE_HPP

#include <vector>
#include <iostream>

namespace veritas {

    /**
     * Store immutable dynamically-sized arrays of type T in stable memory.
     * Pointers returned by `BlockStore::save` are stable for as long as this
     * object lives.
     */
    template <typename T>
    class BlockStore {
        using Block = std::vector<T>;
        std::vector<Block> blocks_;

        Block& get_block_with_remaining_capacity(size_t cap, size_t rem_memory_capacity)
        {
            Block& block = blocks_.back();
            if (block.capacity() - block.size() < cap) // allocate a new block
            {
                size_t rem_capacity = rem_memory_capacity / sizeof(T);
                if (rem_capacity > block.capacity() * 2) // double size of blocks each time,..
                    rem_capacity = block.capacity() * 2; // .. unless memory limit almost reached
                else if (rem_capacity > 0)
                    std::cerr << "WARNING: almost running out of memory, "
                        << static_cast<double>(rem_memory_capacity) / (1024.0*1024.0)
                        << " mb left " << std::endl;
                else
                    throw std::runtime_error("BlockStore: out of memory");

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
            const T *begin;
            const T *end;
        };

        const static size_t MIN_BLOCK_SIZE = 5*1024*1024 / sizeof(T); // 5MB of domains

        BlockStore()
        {
            Block block;
            block.reserve(MIN_BLOCK_SIZE);
            blocks_.push_back(std::move(block));
        }

        // disallow: references change wrt other blockstore, most likely this is a mistake
        BlockStore(const BlockStore&) = delete;
        BlockStore& operator=(const BlockStore&) = delete;
        BlockStore(BlockStore&& o) { std::swap(blocks_, o.blocks_); }
        BlockStore& operator=(BlockStore&& o) { std::swap(blocks_, o.blocks_); return *this; }

        size_t get_mem_size() const
        {
            size_t mem = 0;
            for (const Block& b : blocks_)
                mem += b.capacity() * sizeof(T);
            return mem;
        }

        size_t get_used_mem_size() const
        {
            size_t mem = 0;
            for (const Block& b : blocks_)
                mem += b.size() * sizeof(T);
            return mem;
        }

        template <typename IT>
        Ref store(IT begin, IT end, size_t rem_memory_capacity)
        {
            // this store_ block has enough space to accomodate the workspace DomainBox
            size_t size = end - begin;
            Block& block = get_block_with_remaining_capacity(size, rem_memory_capacity);

            // push a copy of the workspace DomainBox
            size_t start_index = block.size();
            for (; begin != end; ++begin)
                block.push_back(*begin);

            const T *ptr = &block[start_index];

            return { ptr, ptr + size };
        }

        template <typename Container>
        Ref store(const Container& c, size_t rem_memory_capacity)
        { return store(c.begin(), c.end(), rem_memory_capacity); }
    };
} // namespace veritas

#endif // VERITAS_BLOCK_STORE_HPP
