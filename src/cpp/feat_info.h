/*
 * Copyright 2019 DTAI Research Group - KU Leuven.
 * License: Apache License 2.0
 * Author: Laurens Devos
 *
 * ----
 *
 * This file contains reimplemplementations of concepts introduced by the
 * following paper Chen et al. 2019:
 *
 * https://papers.nips.cc/paper/9399-robustness-verification-of-tree-based-models
 * https://github.com/chenhongge/treeVerification
 */

#include <vector>
#include <unordered_map>
#include "tree.h"

#ifndef TREECK_FEAT_INFO_H
#define TREECK_FEAT_INFO_H

namespace treeck {

    class FeatInfo {
    public:
        const int UNUSED_ID = -1;

    private:
        std::vector<FeatId> feat_ids_;
        std::vector<int> id_map_;
        std::vector<bool> is_real_;
        int max_id_;

    public:
        FeatInfo();
        FeatInfo(const AddTree& at);
        FeatInfo(const FeatInfo& instance0,
                 const AddTree& at,
                 const std::unordered_set<FeatId>& matches,
                 bool match_is_reuse);

        int get_id(int feat_id) const;
        int get_max_id() const;
        bool is_id_reused(const FeatInfo& instance0, int feat_id) const;

        bool is_real(FeatId feat_id) const;

        const std::vector<FeatId> feat_ids() const;
    };

} /* namespace treeck */

#endif // TREECK_FEAT_INFO_H
