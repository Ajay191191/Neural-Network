package com.neuralnet.algorithm;

import java.util.ArrayList;
import java.util.List;

import com.neuralnet.NeuralNetwork;
import com.neuralnet.Neuron;
import com.neuralnet.NeuronLayer;

public class Backpropagation implements Algorithm{
	
	private NeuralNetwork network;
	private List<Double> expectedOutput;
	
	private double errorThreshold;
	private double learningRate;

	public double getErrorThreshold() {
		return errorThreshold;
	}

	public void setErrorThreshold(double errorThreshold) {
		this.errorThreshold = errorThreshold;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public NeuralNetwork getNetwork() {
		return network;
	}

	public void setNetwork(NeuralNetwork network) {
		this.network = network;
	}

	public List<Double> getExpectedOutput() {
		return expectedOutput;
	}

	public void setExpectedOutput(List<Double> expectedOutput) {
		this.expectedOutput = expectedOutput;
	}
	
	public Backpropagation(NeuralNetwork network, List<Double> expectedOutput) {
		super();
		this.network = network;
		this.expectedOutput = expectedOutput;
	}

	
	
	public Backpropagation(NeuralNetwork network) {
		super();
		this.network = network;
	}

	public double train(){
		network.compute();
		List<Double> outputs = network.getOutput();
		if(outputs.size() != this.getExpectedOutput().size())
			return 0;
		int numOfLayers = network.getNeuronLayers().size();
		NeuronLayer previousLayer=null;
		Double error = 0.0;
		do{
			for(int layerNumber = numOfLayers-1 ; layerNumber>=0;layerNumber--){
				NeuronLayer layer = network.getNeuronLayer(layerNumber);
				if(layer.isOutputLayer()){
					for(int neuronIndex =0 ;neuronIndex<layer.getNeurons().size();neuronIndex++){
						Neuron neuron = layer.getNeurons().get(neuronIndex);
						if(neuron.isBiasNeuron())
							continue;
						neuron.setDelta(this.getExpectedOutput().get(neuronIndex) - neuron.getOutput() );
					}
					previousLayer= layer;
					continue;
				}
				else{
					for(int neuronIndex =0 ;neuronIndex<layer.getNeurons().size();neuronIndex++){
						Neuron neuron = layer.getNeurons().get(neuronIndex);
						if(neuron.isBiasNeuron())
							continue;
						List<Double> weightForNeuronIndex = previousLayer.getWeightFromIndex(neuronIndex);
						double deltaForNeuron = 0;
						for(int neuronIndexPrevious =1 ; neuronIndexPrevious<previousLayer.getNeurons().size();neuronIndexPrevious++){
							deltaForNeuron += previousLayer.getNeuron(neuronIndexPrevious).getDelta() * weightForNeuronIndex.get(neuronIndexPrevious-1); 
						}
						neuron.setDelta(deltaForNeuron);
					}
				}
				
			}
			
			assignNewWeights(network);
			network.compute();
			outputs=network.getOutput();
			error = getError(outputs, expectedOutput);
			
		}while(error>this.getErrorThreshold());
		return error;
	}
	
	private void assignNewWeights(NeuralNetwork network){
		for(int layerIndex=0;layerIndex<network.getNeuronLayers().size();layerIndex++){
			NeuronLayer layer= network.getNeuronLayer(layerIndex);
			if(layer.isInputLayer())
				continue;
			
			NeuronLayer previousLayer = network.getNeuronLayer(layerIndex-1);
			for(Neuron neuron : layer.getNeurons()){
				if(neuron.isBiasNeuron())
					continue;
				List<Double> weights = neuron.getWeights();
				List<Double> newWeights = new ArrayList<Double>();
				for(int weightIndex = 0 ;weightIndex<weights.size();weightIndex++){
					Neuron neuronForWeight = previousLayer.getNeuron(weightIndex);
					Double weight = weights.get(weightIndex);
					weight += neuronForWeight.getOutput()*this.getLearningRate()*neuron.getDelta()*neuron.computeOutput()*(1-neuron.computeOutput());
					newWeights.add(weight);
				}
				neuron.setWeights(newWeights);
			}
			
		}
	}
	
	public Double getError(List<Double> actual, List<Double> expected) {
		
		List<Double> actual1 = new ArrayList<Double>();
		actual1.addAll(actual);
		actual1.remove(0);
		List<Double> expected1 = new ArrayList<Double>();
		expected1.addAll(expected);
		expected1.remove(0);
		
        if (actual1.size()!= expected1.size()) {
        	return null;
        }

        double sum = 0;

        for (int i = 0; i < expected1.size(); i++) {
            sum += Math.pow(expected1.get(i) - actual1.get(i), 2);
        }

        return sum / 2;
    }

}
