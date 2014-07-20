package com.neuralnet.utility;


import java.util.List;

public class WeightsFileReader{
	
	private int layerIndex;
	private int neuronIndex;
	private List<Double> weights;
	
	
	public WeightsFileReader(int layerIndex, int neuronIndex,
			List<Double> weights) {
		super();
		this.layerIndex = layerIndex;
		this.neuronIndex = neuronIndex;
		this.weights = weights;
	}
	public int getLayerIndex() {
		return layerIndex;
	}
	public void setLayerIndex(int layerIndex) {
		this.layerIndex = layerIndex;
	}
	public int getNeuronIndex() {
		return neuronIndex;
	}
	public void setNeuronIndex(int neuronIndex) {
		this.neuronIndex = neuronIndex;
	}
	public List<Double> getWeights() {
		return weights;
	}
	public void setWeights(List<Double> weights) {
		this.weights = weights;
	}
	
}