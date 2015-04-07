package com.neuralnet.algorithm;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

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
		
		Double error = 0.0;
		do{
			IntStream.range(numOfLayers-1, 0).parallel().forEach(new IntConsumer() {
				NeuronLayer previousLayer=null;
				@Override
				public void accept(int layerNumber) {
					NeuronLayer layer = network.getNeuronLayer(layerNumber);
					if(layer.isOutputLayer()){
						IntStream.range(0, layer.getNeurons().size()).filter(neuronIndex -> !layer.getNeurons().get(neuronIndex).isBiasNeuron()).parallel().forEach(neuronIndex->{
							Neuron neuron = layer.getNeurons().get(neuronIndex);
							neuron.setDelta(getExpectedOutput().get(neuronIndex) - neuron.getOutput() );
						});
						previousLayer= layer;
						return;
					}
					else{
						IntStream.range(0, layer.getNeurons().size()).parallel().filter(neuronIndex->!layer.getNeurons().get(neuronIndex).isBiasNeuron()).forEach(neuronIndex->{
							Neuron neuron = layer.getNeurons().get(neuronIndex);
							List<Double> weightForNeuronIndex = previousLayer.getWeightFromIndex(neuronIndex);
							double deltaForNeuron = 0;
							for(int neuronIndexPrevious =1 ; neuronIndexPrevious<previousLayer.getNeurons().size();neuronIndexPrevious++){
								deltaForNeuron += previousLayer.getNeuron(neuronIndexPrevious).getDelta() * weightForNeuronIndex.get(neuronIndexPrevious-1); 
							}
							neuron.setDelta(deltaForNeuron);
						});
					}
				}
			});
			
			assignNewWeights(network);
			network.compute();
			outputs=network.getOutput();
			error = getError(outputs, expectedOutput);
			
		}while(error>this.getErrorThreshold());
		return error;
	}
	
	private void assignNewWeights(NeuralNetwork network){
		IntStream.range(0, network.getNeuronLayers().size()).filter(layerIndex->!network.getNeuronLayer(layerIndex).isInputLayer()).forEach(layerIndex->{
			NeuronLayer layer= network.getNeuronLayer(layerIndex);
			NeuronLayer previousLayer = network.getNeuronLayer(layerIndex-1);
			layer.getNeurons().parallelStream().filter(neuron->!neuron.isBiasNeuron()).forEach(neuron->{
				List<Double> weights = neuron.getWeights();
				List<Double> newWeights = new ArrayList<Double>();
				IntStream.range(0, weights.size()).parallel().forEach(weightIndex->{
					Neuron neuronForWeight = previousLayer.getNeuron(weightIndex);
					Double weight = weights.get(weightIndex);
					weight += neuronForWeight.getOutput()*this.getLearningRate()*neuron.getDelta()*neuron.computeOutput()*(1-neuron.computeOutput());
					newWeights.add(weight);
				});
				neuron.setWeights(newWeights);
			});
		});
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
