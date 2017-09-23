#pragma once

#include <sparse_map/common.h>

namespace NSparseMap {

	class Model {
	public:
		Model(ui32 batchSize, ui32 size);

		

	private:
		TMatrix Membrane;
	};




} // NSparseMap

