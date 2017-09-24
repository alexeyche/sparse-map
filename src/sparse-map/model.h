#pragma once

#include <sparse-map/common.h>

namespace NSparseMap {

	class TModel {
	public:
		TModel(ui32 batchSize, ui32 inputSize, ui32 filterSize, ui32 layerSize);


		void Tick(const TMatrix& input);
		

	private:
		ui32 BatchSize;
		ui32 InputSize;
		ui32 FilterSize;
		ui32 LayerSize;

	public:
		TMatrix Membrane;
	};




} // NSparseMap

