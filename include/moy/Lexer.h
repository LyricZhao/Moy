#pragma once

#include <memory>
#include <string>

namespace moy {

/// Structure definition a location in a file.
struct Location {
    std::shared_ptr<std::string> file; ///< filename.
    int line;                          ///< line number.
    int col;                           ///< column number.
};

}
