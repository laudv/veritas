#ifndef TREECK_PRUNE_H
#define TREECK_PRUNE_H

#include <vector>

#include "domain.h"
#include "tree.hpp"

namespace treeck {

    AddTree prune(const AddTree& addtree, const Domains& domains);
    AddTree::TreeT prune(const AddTree::TreeT& addtree, const Domains& domains);

} /* namespace treeck */

#endif /* TREECK_PRUNE_H */
