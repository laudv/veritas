#ifndef TREECK_ADDTREE_H
#define TREECK_ADDTREE_H

#include "tree.h"

namespace treeck {

    class AddTree {
    public:
        using TreeT = Tree<double>;

    private:
        std::vector<TreeT> trees;

    public:
        double base_score;

        AddTree();

        size_t add_tree(TreeT&& tree);
        size_t size() const;

        TreeT& operator[](size_t index);
        const TreeT& operator[](size_t index) const;

        std::string to_json();
        static AddTree from_json(const std::string& json);
    };

} /* namespace treeck */

#endif /* TREECK_ADDTREE_H */
