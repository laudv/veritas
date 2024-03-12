/*
 * Copyright 2023 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_FEAT_MAP_HPP
#define VERITAS_FEAT_MAP_HPP

#include "basics.hpp"
#include "tree.hpp"
#include "addtree.hpp"

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <functional>

namespace veritas {

    /**
     * A dataset has a list of features, usually with names.
     * Internally, an index is used to refer to a particular feature, going
     * from 0 for the first feature name, and len(features) for the last
     * feature name.
     *
     * TODO write tests for this
     * TODO split interface and impl / template tree
     */
    class FeatMap {
        std::vector<std::string> names_;
        std::map<std::reference_wrapper<const std::string>, FeatId, std::less<const std::string>> index_map_;

        // disjoint set data structure mapping index -> feat id
        // vector is twice the length of names_, first set for first instance, second set for second instance
        mutable std::vector<FeatId> feat_ids_;

        using IterValueType = std::tuple<const std::string&, FeatId>;

    public:

        template <typename ...Args>
        explicit
        FeatMap(Args... feature_names) : names_(feature_names...) { init(); }

        FeatMap(FeatId num_features) {
            for (FeatId index = 0; index < num_features; ++index)
            {
                std::stringstream buffer;
                buffer << "feature" << index;
                names_.push_back(buffer.str());
            }
            init();
        }

        size_t num_features() const { return names_.size(); }

        FeatId get_index(const std::string& feature_name, int instance) const {

            auto it = index_map_.find(feature_name);
            if (it != index_map_.end())
                return get_index(it->second, instance);
            throw std::runtime_error("invalid feature name");
        }

        FeatId get_index(FeatId index, int instance) const {
            // second instance's indexes are offset by number of features
            int nfeats = static_cast<int>(num_features());
            int offset = clean_instance(instance) * nfeats;
            return index + offset;
        }

        int get_instance(FeatId index) const {
            return static_cast<size_t>(index) >= num_features();
        }

        const std::string &get_name(FeatId index) const {
            return names_.at(index % num_features());
        }

        void get_indices_map(std::multimap<FeatId, FeatId>& out, int instance=-1) const {
            FeatId begin = (instance==1) * static_cast<FeatId>(num_features());
            FeatId end = (static_cast<FeatId>(instance!=0) + 1) * static_cast<FeatId>(num_features());

            for (FeatId index = begin; index < end; ++index) {
                FeatId feat_id = get_feat_id(index);
                out.insert({feat_id, index});
            }
        }

        std::multimap<FeatId, FeatId> get_indices_map(int instance=-1) const {
            std::multimap<FeatId, FeatId> mmap;
            get_indices_map(mmap, instance);
            return mmap;
        }

        void share_all_features_between_instances() {
            for (FeatId index = 0; static_cast<size_t>(index) < names_.size(); ++index)
                uf_union(index, index+num_features());
        }

        FeatId get_feat_id(FeatId index) const { return uf_find(index); } 
        FeatId get_feat_id(const std::string& feat_name, int instance=0) const
        { return get_feat_id(get_index(feat_name, instance)); }

        void use_same_id_for(FeatId index1, FeatId index2) { uf_union(index1, index2); }

        /** Replace the feature ids used in the given at by the replacements in this FeatMap. */
        AddTree transform(const AddTree& at, int instance=0) const {
            instance = clean_instance(instance);
            AddTree new_at(at.num_leaf_values(), at.get_type());
            for (const Tree& t : at) {
                Tree& new_t = new_at.add_tree();
                transform(t, t.root(), new_t, new_t.root(), instance);
            }

            return new_at;
        }

        class iterator { // simple range iter over indexes
            friend FeatMap;
            FeatId index;

        public:
            iterator(FeatId index) : index(index) {}
            iterator& operator++() { ++index; return *this; }
            iterator operator++(int) { iterator retval = *this; ++(*this); return retval; }
            bool operator==(iterator other) const { return index == other.index; }
            bool operator!=(iterator other) const { return !(*this == other); }
            FeatId operator*() { return index; }

            // iterator traits
            using value_type = FeatId;
            using difference_type = size_t;
            using pointer = const FeatId*;
            using reference = const FeatId&;
            using iterator_category = std::forward_iterator_tag;
        };

        iterator begin(int instance = 0) const {
            return {static_cast<FeatId>(clean_instance(instance) * num_features())};
        }

        iterator end(int instance = 1) const {
            return {static_cast<FeatId>((1+clean_instance(instance)) * num_features())};
        }

        struct instance_iter_helper {
            FeatId begin_, end_;

            iterator begin() const { return {begin_}; }
            iterator end() const { return {end_}; }
        };

        instance_iter_helper iter_instance(int instance) const {
            return { begin(instance).index, end(instance).index };
        }

    private:

        // https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        FeatId uf_find(FeatId index) const {
            while (feat_ids_[index] != index) {
                feat_ids_[index] = feat_ids_[feat_ids_[index]];
                index = feat_ids_[index];
            }
            return index;
        }

        void uf_union(FeatId index1, FeatId index2) {
          index1 = uf_find(index1);
          index2 = uf_find(index2);

          if (index1 == index2)
            return;
          if (index1 > index2)
            std::swap(index1, index2);

          feat_ids_[index2] = index1;
        }

        int clean_instance(int instance) const { return std::min(1, std::max(0, instance)); }

        void init() {
            for (const std::string& name : names_) {
                index_map_.insert({name, feat_ids_.size()});
                feat_ids_.push_back(feat_ids_.size());
            }

            // add len(names) to feat_ids_ for second instance
            for (size_t i = 0; i < names_.size(); ++i)
                feat_ids_.push_back(feat_ids_.size());

        }

        /**
         * Replace the feat_ids in the given AddTree.
         *
         * Assumed is that the AddTree uses indexes as feature_ids not yet
         * offset for instance0 or instance1.
         */
        void transform(const Tree& tn, NodeId n, Tree& tm, NodeId m, int instance) const {
            if (tn.is_internal(n)) {
                LtSplit s = tn.get_split(n);
                FeatId index = get_index(s.feat_id, instance);
                if (static_cast<size_t>(index) >= feat_ids_.size())
                    throw std::runtime_error("feature index out of bounds");
                FeatId feat_id = get_feat_id(index);
                tm.split(m, {feat_id, s.split_value});
                transform(tn, tn.right(n), tm, tm.right(m), instance);
                transform(tn, tn.left(n), tm, tm.left(m), instance);
            } else {
                for (int i = 0; i < tn.num_leaf_values(); ++i)
                    tm.leaf_value(m, i) = tn.leaf_value(n, i);
            }
        }
    }; // class FeatMap

    inline std::ostream& operator<<(std::ostream& s, const FeatMap& fm) {
        s << "FeatMap {" << std::endl;
        for (auto index : fm)
            s << "    [" << index << "] `" << fm.get_name(index)
              << "` -> " << fm.get_feat_id(index)
              << " (instance " << fm.get_instance(index) << ')' << std::endl;
        s << '}';
        return s;
    }
} // namespace veritas

#endif // VERITAS_FEAT_MAP_HPP
