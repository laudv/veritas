/*
 * Copyright 2020 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
*/

#ifndef VERITAS_FEAT_MAP_HPP
#define VERITAS_FEAT_MAP_HPP

#include "basics.hpp"
#include "new_tree.hpp"

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
     * 
     *
     */
    class FeatMap {
        const std::vector<std::string> names_;
        std::map<std::reference_wrapper<const std::string>, FeatId, std::less<const std::string>> index_map_;

        // disjoint set data structure mapping index -> feat id
        // vector is twice the length of names_, first set for first instance, second set for second instance
        mutable std::vector<FeatId> feat_ids_;

        using IterValueType = std::tuple<const std::string&, FeatId>;

    public:

        template <typename ...Args>
        explicit
        FeatMap(Args... feature_names) : names_(feature_names...)
        {
            for (const std::string& name : names_)
            {
                index_map_.insert({name, feat_ids_.size()});
                feat_ids_.push_back(feat_ids_.size());
            }

            // add len(names) to feat_ids_ for second instance
            for (size_t i = 0; i < names_.size(); ++i)
                feat_ids_.push_back(feat_ids_.size());
        }

        size_t num_features() const { return names_.size(); }

        FeatId get_index(const std::string& feature_name, int instance) const
        {
            // second instance's indexes are offset by number of features
            int offset = clean_instance(instance) * num_features();

            auto it = index_map_.find(feature_name);
            if (it != index_map_.end())
                return it->second + offset;

            return -1;
        }

        int get_instance(FeatId index) const
        {
            return static_cast<size_t>(index) > num_features();
        }

        void share_all_features_between_instances()
        {
            for (FeatId index = 0; static_cast<size_t>(index) < names_.size(); ++index)
                uf_union(index, index+num_features());
        }

        FeatId get_feat_id(FeatId index) const { return uf_find(index); } 
        FeatId get_feat_id(const std::string& feat_name, int instance=0) const
        { return get_feat_id(get_index(feat_name, instance)); }

        void use_same_id_for(FeatId index1, FeatId index2) { uf_union(index1, index2); }

        AddTree transform(const AddTree& at, int instance=0) const
        {
            // TODO
            return {};
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

        iterator begin(int instance = 0) const
        {
            return {static_cast<FeatId>(clean_instance(instance) * num_features())};
        }

        iterator end(int instance = 1) const
        {
            return {static_cast<FeatId>((1+clean_instance(instance)) * num_features())};
        }

        struct instance_iter_helper {
            FeatId begin_, end_;

            iterator begin() const { return {begin_}; }
            iterator end() const { return {end_}; }
        };

        instance_iter_helper iter_instance(int instance) const
        {
            return { begin(instance).index, end(instance).index };
        }

    private:

        // https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        FeatId uf_find(FeatId index) const
        {
            // function Find(x) is
            //     while x.parent ≠ x do
            //         x.parent := x.parent.parent
            //         x := x.parent
            //     end while
            //     return x
            // end function

            while (feat_ids_[index] != index)
            {
                feat_ids_[index] = feat_ids_[feat_ids_[index]];
                index = feat_ids_[index];
            }
            return index;
        }

        void uf_union(FeatId index1, FeatId index2) {
          // function Union(x, y) is
          //     // Replace nodes by roots
          //     x := Find(x)
          //     y := Find(y)

          //     if x = y then
          //         return  // x and y are already in the same set
          //     end if

          //     // If necessary, rename variables to ensure that
          //     // x has at least as many descendants as y
          //     if x.size < y.size then
          //         (x, y) := (y, x)
          //     end if

          //     // Make x the new root
          //     y.parent := x
          //     // Update the size of x
          //     x.size := x.size + y.size
          // end function

          index1 = uf_find(index1);
          index2 = uf_find(index2);

          if (index1 == index2)
            return;
          if (index1 > index2)
            std::swap(index1, index2);

          feat_ids_[index2] = index1;
        }

        int clean_instance(int instance) const { return std::min(1, std::max(0, instance)); }
    };

    //std::ostream& operator<<(std::ostream& s, const FeatMap& fm)
    //{
    //    s << "FeatMap {" << std::endl;
    //    for (auto&& [f, id] : fm)
    //        std::cout << "   - `" << f << "` -> " << id << std::endl;
    //    s << '}';
    //    return s;
    //}
}

#endif // VERITAS_FEAT_MAP_HPP
