#include "model.h"

namespace NSparseMap {


	TModel::TModel(ui32 batchSize, ui32 inputSize, ui32 filterSize, ui32 layerSize)
		: BatchSize(batchSize)
		, InputSize(inputSize)
		, FilterSize(filterSize)
		, LayerSize(layerSize)
		, Membrane(TMatrix::Zero(batchSize, layerSize))
	{
	}


	void TModel::Tick(const TMatrix& input) {
		

		
	}

} // NSparseMap
