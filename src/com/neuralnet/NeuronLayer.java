package com.neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import com.neuralnet.utility.NeuralNetUtility;
import com.neuralnet.utility.WeightsFileReader;

public class NeuronLayer implements NeuralNetBase{

	private int numberOfNeurons;
	private List<Neuron> neurons;
	private List<Double> previousOutputs;

	private boolean isInputLayer;
	private boolean isOutputLayer;
	private int layerIndex;

	public boolean isInputLayer() {
		return isInputLayer;
	}

	public void setInputLayer(boolean isInputLayer) {
		this.isInputLayer = isInputLayer;
	}

	public boolean isOutputLayer() {
		return isOutputLayer;
	}

	public void setOutputLayer(boolean isOutputLayer) {
		this.isOutputLayer = isOutputLayer;
	}

	public int getNumberOfNeurons() {
		return numberOfNeurons;
	}

	public void setNumberOfNeurons(int numberOfNeurons) {
		this.numberOfNeurons = numberOfNeurons;
	}

	public List<Neuron> getNeurons() {
		return neurons;
	}

	public void setNeurons(List<Neuron> neurons) {
		this.neurons = neurons;
	}

	public NeuronLayer(int numberOfNeurons, int numInputs, int layerIndex) {
		super();
		this.numberOfNeurons = numberOfNeurons;
		this.neurons = new ArrayList<Neuron>();
		this.setLayerIndex(layerIndex);
		IntStream.range(0, this.numberOfNeurons + 1).forEach(i -> {
			WeightsFileReader reader = NeuralNetUtility.getReader(this.layerIndex, i);
			if (reader != null) {
				Neuron e = new Neuron(numInputs, reader.getWeights());
				if (i == 0) {
					e.setBiasNeuron(true);
				}
				this.neurons.add(e);
			} else {
				Neuron e = new Neuron(numInputs);
				if (i == 0) {
					e.setBiasNeuron(true);
				}
				this.neurons.add(e);
			}
		});
	}

	public NeuronLayer(int numberOfNeurons, int numInputs, List<Double> weights) {
		super();
		this.numberOfNeurons = numberOfNeurons;
		this.neurons = new ArrayList<Neuron>();
		IntStream.range(0, this.numberOfNeurons + 1).parallel().forEach(i->{
			Neuron e = new Neuron(numInputs, weights);
			if (i == 0)
				e.setBiasNeuron(true);
			this.neurons.add(e);
		});
	}

	public NeuronLayer(int numberOfNeurons, List<Neuron> neurons) {
		super();
		this.numberOfNeurons = numberOfNeurons;
		this.neurons = neurons;
	}

	public Neuron getNeuron(int index) {
		if (this.neurons != null)
			return this.neurons.get(index);
		return null;
	}

	public void compute() {

		this.neurons.parallelStream().filter(neuron->!neuron.isBiasNeuron()).forEach(neuron->{
			List<Double> weights = neuron.getWeights();
			List<Double> newWeights = new ArrayList<Double>(weights.size());
			IntStream.range(0, weights.size()).parallel().forEach(i -> {
				newWeights.add(weights.get(i)* this.getPreviousOutputs().get(i));
			});
			neuron.setNewWeights(newWeights);
			neuron.compute();
		});
	}

	public List<Double> getPreviousOutputs() {
		return previousOutputs;
	}

	public void setPreviousOutputs(List<Double> previousOutputs) {
		this.previousOutputs = previousOutputs;
	}

	public int getLayerIndex() {
		return layerIndex;
	}

	public void setLayerIndex(int layerIndex) {
		this.layerIndex = layerIndex;
	}

	public List<Double> getWeightFromIndex(int index) {
		List<Double> weightsFromIndex = new ArrayList<Double>();
		for (Neuron neuron : this.neurons) {
			if(!neuron.isBiasNeuron())
				weightsFromIndex.add(neuron.getWeight(index));
		}
		return weightsFromIndex;
	}
	
	public List<Double> getDeltas(){
		List<Double> deltas = new ArrayList<Double>();
		for(Neuron neuron : this.neurons){
			if(!neuron.isBiasNeuron())
				deltas.add(neuron.getDelta());
		}
		return deltas;
	}

	@Override
	public String toString() {
		return "NeuronLayer [neurons=" + neurons + "]";
	}
	
}
