/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_DOMAIN_STORE_HPP
#define VERITAS_DOMAIN_STORE_HPP

#include "new_tree.hpp"
#include <algorithm>
#include <iostream>

namespace veritas {

    /**
     * The search generates many states. Organize memory in big Blocks for
     * efficiency.
     * Pointers into the DomainStore must be stable -> no vector resizing!.
     *
     * Everything in the `DomainStore` is immutable.
     */
    class DomainStore {
        using Block = std::vector<DomainPair>;
        std::vector<Block> store_;

        Block& get_block_with_remaining_capacity(size_t cap)
        {
            Block& block = store_.back();
            if (block.capacity() - block.size() < cap) // allocate a new block
            {
                size_t mem = get_mem_size();
                size_t rem_capacity = (max_mem_size - mem) / sizeof(Block::value_type);
                if (rem_capacity > block.capacity() * 2) // double size of blocks each time,..
                    rem_capacity = block.capacity() * 2; // .. unless memory limit almost reached
                else if (rem_capacity > 0)
                    std::cerr << "WARNING: almost running out of memory, "
                        << static_cast<double>(rem_capacity * sizeof(Block::value_type)) / (1024.0*1024.0)
                        << " mb out of "
                        << static_cast<double>(max_mem_size) / (1024.0*1024.0)
                        << " mb left" << std::endl;
                else
                    throw std::runtime_error("DomainStore: out of memory");

                Block new_block;
                new_block.reserve(rem_capacity);
                store_.push_back(std::move(new_block));

                //std::cout << "DomainStore memory: " << mem << " bytes, "
                //    << (static_cast<double>(get_mem_size()) / (1024.0 * 1024.0)) << " mb ("
                //    << store_.size() << " blocks)" << std::endl;
            }

            return store_.back();
        }

    public:
        const static size_t MIN_BLOCK_SIZE = 5*1024*1024 / sizeof(Block::value_type); // 5MB of domains
        size_t max_mem_size = size_t(4)*1024*1024*1024;

        DomainStore()
        {
            Block block;
            block.reserve(MIN_BLOCK_SIZE);
            store_.push_back(std::move(block));
        }

        size_t get_mem_size() const
        {
            size_t mem = 0;
            for (const Block& b : store_)
                mem += b.capacity() * sizeof(Block::value_type);
            return mem;
        }

        size_t get_used_mem_size() const
        {
            size_t mem = 0;
            for (const Block& b : store_)
                mem += b.size() * sizeof(Block::value_type);
            return mem;
        }

        //void refine_workspace(LtSplit split, bool from_left_child)
        //{
        //    Domain dom = from_left_child
        //        ? std::get<0>(split.get_domains())
        //        : std::get<1>(split.get_domains());
        //    refine_domains(workspace_, split.feat_id, dom);
        //}
//
//        Box get_workspace_box() const { return Box(workspace_); }
//
//        Box push_workspace()
//        {
//            // this store_ block has enough space to accomodate the workspace DomainBox
//            Box workspace = get_workspace_box();
//            Block& block = get_block_with_remaining_capacity(workspace.size());
//
//            // push a copy of the workspace DomainBox
//            size_t start_index = block.size();
//            for (auto&& [id, domain] : workspace)
//                block.push_back({ id, domain });
//
//            DomainPair *ptr = &block[start_index];
//            Box box = { ptr, ptr + workspace_.size() };
//
//            workspace_.clear();
//
//            return box;
//        }

        //void combine_in_workspace(const BoxRef& a, const BoxRef& b, bool copy_b)
        //{
        //    if (!workspace_.empty())
        //        throw std::runtime_error("workspace not empty");

        //    const DomainPair *it0 = a.begin();
        //    const DomainPair *it1 = b.begin();

        //    // assume sorted
        //    while (it0 != a.end() && it1 != b.end())
        //    {
        //        if (it0->feat_id == it1->feat_id)
        //        {
        //            Domain dom = it0->domain.intersect(it1->domain);
        //            workspace_.push_back({ it0->feat_id, dom });
        //            ++it0; ++it1;
        //        }
        //        else if (it0->feat_id < it1->feat_id)
        //        {
        //            workspace_.push_back(*it0); // copy
        //            ++it0;
        //        }
        //        else
        //        {
        //            if (copy_b)
        //                workspace_.push_back(*it1); // copy
        //            ++it1;
        //        }
        //    }

        //    // push all remaining items (one of them is already at the end, no need to compare anymore)
        //    for (; it0 != a.end(); ++it0)
        //        workspace_.push_back(*it0); // copy
        //    for (; copy_b && it1 != b.end(); ++it1)
        //        workspace_.push_back(*it1); // copy
        //}

        //Box combine_and_push(const Box& a, const Box& b, bool copy_b)
        //{
        //    combine_in_workspace(a, b, copy_b);
        //    return push_workspace();
        //}

        //void clear_workspace()
        //{
        //    workspace_.clear();
        //}

    };

} // namespace veritas


#endif // VERITAS_DOMAIN_STORE_HPP
