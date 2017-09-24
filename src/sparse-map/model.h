#pragma once

#include <sparse-map/common.h>

namespace NSparseMap {

	inline TMatrix unitScalingInit(ui32 x, ui32 y) {
		TMatrix m = TMatrix::Random(x, y);
		double initSpan = std::sqrt(3.0)/std::sqrt(std::max(x, y));
		return 2.0 * initSpan * (m.array() + 1.0) / 2.0 - initSpan;
	}


	inline TMatrix relu(TMatrix x) {
		return x.cwiseMax(0.0);
	}

	class TModel {
	public:
		TModel(
			ui32 batchSize, 
			ui32 inputSize, 
			ui32 filterSize, 
			ui32 layerSize, 
			double tau,
			double lambda
		)
			: BatchSize(batchSize)
			, InputSize(inputSize)
			, FilterSize(filterSize)
			, LayerSize(layerSize)
			, Tau(tau)
			, Lambda(lambda)
			, Membrane(TMatrix::Zero(batchSize, layerSize))
			, Activation(TMatrix::Zero(batchSize, layerSize))
		{
			F = unitScalingInit(FilterSize * InputSize, LayerSize);
			F = (F.array().rowwise())/(F.colwise().norm().array());
			Fc = F.transpose() * F - TMatrix::Identity(LayerSize, LayerSize);
		}

		void Tick(const TMatrix& input) {
			TMatrix feedback = Activation * Fc;
			
			TMatrix dMem = - Membrane + input * F - feedback;

			Membrane += dMem / Tau;

			Activation = relu(Membrane.array() - Lambda);
		}
		

	private:
		ui32 BatchSize;
		ui32 InputSize;
		ui32 FilterSize;
		ui32 LayerSize;
		double Tau;
		double Lambda;

	public:
		TMatrix Membrane;
		TMatrix Activation;
		
		TMatrix F;
		TMatrix Fc;
	};




} // NSparseMap

