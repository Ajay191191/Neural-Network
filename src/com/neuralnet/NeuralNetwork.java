package com.neuralnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.neuralnet.utility.NeuralNetUtility;

public class NeuralNetwork implements NeuralNetBase{
	
	private int numInputs;
	private int numOutputs;
	private List<NeuronLayer> neuronLayers;	
	
	public int getNumInputs() {
		return numInputs;
	}
	public void setNumInputs(int numInputs) {
		this.numInputs = numInputs;
	}
	public int getNumOutputs() {
		return numOutputs;
	}
	public void setNumOutputs(int numOutputs) {
		this.numOutputs = numOutputs;
	}
	public List<NeuronLayer> getNeuronLayers() {
		return neuronLayers;
	}
	public void setNeuronLayers(List<NeuronLayer> neuronLayers) {
		this.neuronLayers = neuronLayers;
	}
	
	public NeuralNetwork(int numInputs,int numOutputs) {
		this.numInputs = numInputs;
		this.numOutputs = numOutputs;
	}
	
	public NeuronLayer getNeuronLayer(int index){
		if(this.neuronLayers!=null && !this.neuronLayers.isEmpty())
			return this.neuronLayers.get(index);
		return null;
	}
	
	
	public void createNet(List<Integer> layersNodes,String weights){
		NeuralNetUtility.readWeights(weights);
		neuronLayers = new ArrayList<NeuronLayer>();
		
		for(int i=0;i<layersNodes.size();i++){
			NeuronLayer layer = null;
			if(i==0){
				layer = new NeuronLayer(layersNodes.get(i), layersNodes.get(i),i);
				layer.setInputLayer(true);
			}
			else{
				
				layer = new NeuronLayer(layersNodes.get(i), layersNodes.get(i-1)+1,i);
			}
			if(i==layersNodes.size()-1)
				layer.setOutputLayer(true);
			this.neuronLayers.add(layer);
		}
		
		
	}
	
	public void createNet(List<Integer> layersNodes, List<Double> inputs,String weightsFileName){
		
		NeuralNetUtility.readWeightsFile(weightsFileName);
		
		neuronLayers = new ArrayList<NeuronLayer>();
		if(inputs.size() != numInputs)
			return;
		for(int i=0;i<layersNodes.size();i++){
			NeuronLayer layer = null;
			if(i==0){
				layer = new NeuronLayer(layersNodes.get(i), layersNodes.get(i),i);
				layer.setInputLayer(true);
				int index=0;
				for(Neuron neuron:layer.getNeurons()){
					neuron.setWeights(new ArrayList<Double>(Arrays.asList(inputs.get(i))));
					if(!neuron.isBiasNeuron()){
						neuron.setOutput(inputs.get(index++));
					}
				}
			}
			else{
				
				layer = new NeuronLayer(layersNodes.get(i), layersNodes.get(i-1)+1,i);
			}
			if(i==layersNodes.size()-1)
				layer.setOutputLayer(true);
			this.neuronLayers.add(layer);
		}
	}
	
	
	public void compute(){
		
		for(int i=0;i<this.neuronLayers.size();i++){
			NeuronLayer layer = this.neuronLayers.get(i);
			
			if(layer.isInputLayer())
				continue;
			
			List<Double> previousOutputs = new ArrayList<Double>();
			
			NeuronLayer previousLayer = this.neuronLayers.get(i-1);
			for(Neuron neuron:previousLayer.neurons){
				previousOutputs.add(neuron.getOutput());
			}
			
			layer.setPreviousOutputs(previousOutputs);
			layer.compute();
		}
		
	}
	
	public List<Double> getOutput(){
		List<Double> output = new ArrayList<Double>();
		List<Neuron> neurons = this.neuronLayers.get(this.neuronLayers.size()-1).getNeurons();
		for(Neuron neuron : neurons){
			output.add(neuron.getOutput());
		}
		return output;
	}
	
	public void setInput(List<Double> inputs){
		NeuronLayer layer = this.getInputLayer();
		if(layer.getNeurons().size() -1!= inputs.size())
			return;
		for(int inputIndex=0;inputIndex<inputs.size();inputIndex++){
			layer.getNeuron(inputIndex+1).setOutput(inputs.get(inputIndex));
		}
	}
	
	public NeuronLayer getInputLayer(){
		for(NeuronLayer layer : this.neuronLayers)
			if(layer.isInputLayer())
				return layer;
		return null;
	}
	
	@Override
	public String toString() {
		return "NeuralNetwork [neuronLayers=" + neuronLayers + "]";
	}

}