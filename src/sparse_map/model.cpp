#include "model.h"

namespace NSparseMap {


	Model::Model(ui32 batchSize, ui32 size)
		: Membrane(batchSize, size)
	{
		std::cout << Membrane;
	}


} // NSparseMap
