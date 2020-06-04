#include "feat_info.h"

namespace treeck {

    FeatInfo::FeatInfo() : max_id_(-1) {}

    FeatInfo::FeatInfo(const AddTree& at) : FeatInfo()
    {
        auto at_splits = at.get_splits();
        int max_feat_id = -1;
        for (auto&& [feat_id, _] : at_splits)
        {
            feat_ids_.push_back(feat_id);
            max_feat_id = std::max(max_feat_id, feat_id);
        }

        std::sort(feat_ids_.begin(), feat_ids_.end());

        id_map_.resize(max_feat_id+1, UNUSED_ID);
        is_real_.resize(max_feat_id+1, false);
        for (auto feat_id : feat_ids_)
        {
            id_map_[feat_id] = ++max_id_;
            auto split_values = at_splits.find(feat_id);
            if (split_values == at_splits.end())
                throw std::runtime_error("hmm, invalid state");
            if (split_values->second.size() != 0) // split values => real split
                is_real_[feat_id] = true;
        }
    }

    FeatInfo::FeatInfo(const FeatInfo& instance0,
                 const AddTree& at,
                 const std::unordered_set<FeatId>& matches,
                 bool match_is_reuse)
        : FeatInfo(at)
    {
        max_id_ = instance0.max_id_;
        for (int i = 0; i < feat_ids_.size(); ++i)
        {
            int feat_id = feat_ids_[i];
            bool in_matches = matches.find(feat_id) != matches.end();
            if (in_matches == match_is_reuse)
                id_map_[feat_id] = instance0.id_map_[feat_id]; // same id as at0!
            else
                id_map_[feat_id] = ++max_id_;
        }
    }

    int
    FeatInfo::get_id(int feat_id) const
    {
        if (feat_id < id_map_.size())
            return id_map_[feat_id];
        return UNUSED_ID;
    }

    int
    FeatInfo::get_max_id() const
    {
        return max_id_;
    }

    bool 
    FeatInfo::is_id_reused(const FeatInfo& instance0, int feat_id) const
    {
        int id0 = instance0.get_id(feat_id);
        int id1 = get_id(feat_id);

        return id0 != UNUSED_ID && id0 != id1;
    }

    const std::vector<FeatId>
    FeatInfo::feat_ids() const
    {
        return feat_ids_;
    }

    bool
    FeatInfo::is_real(FeatId feat_id) const
    {
        return is_real_[feat_id];
    }
}
