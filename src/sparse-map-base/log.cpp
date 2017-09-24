#include "log.h"

namespace NSparseMap {


    TLog& TLog::Instance() {
        static TLog _inst;
        return _inst;
    }


}
