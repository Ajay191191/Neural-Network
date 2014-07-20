package com.neuralnet;

import java.util.ArrayList;
import java.util.List;

import com.neuralnet.utility.Constants;
import com.neuralnet.utility.NeuralNetUtility;

public class Neuron implements NeuralNetBase{
	
	private int numInputs;
	private List<Double> weights;
	private List<Double> newWeights;
	private double output;
	private boolean isBiasNeuron;
	private double delta;
	
	public int getNumInputs() {
		return numInputs;
	}
	public void setNumInputs(int numInputs) {
		this.numInputs = numInputs;
	}
	public List<Double> getWeights() {
		return weights;
	}
	public void setWeights(List<Double> weights) {
		this.weights = weights;
	}
	public Neuron(int numInputs) {
		super();
		this.numInputs = numInputs;
		this.weights = new ArrayList<Double>();
		for(int i=0;i<this.numInputs;i++){
			if(i==0)
				this.getWeights().add(Constants.BIAS);
			else
				this.getWeights().add(Math.random()*10);
		}
	}
	
	public Neuron(int numInputs, List<Double> weights) {
		super();
		this.numInputs = numInputs;
		this.weights = weights;
	}
	
	public double getOutput() {
		return output;
	}
	public void setOutput(double output) {
		this.output = output;
	}
	
	public double computeOutput(){
		double sum=0;
		for (Double weight : newWeights) {
			sum += weight;
		}
		return NeuralNetUtility.sigmoid(sum, Constants.RESPONSE) /*< 0.5 ? 0 : 1*/;
	}
	
	public void compute(){
		this.output = this.computeOutput();
	}
	public boolean isBiasNeuron() {
		return isBiasNeuron;
	}
	public void setBiasNeuron(boolean isBiasNeuron) {
		this.isBiasNeuron = isBiasNeuron;
		this.setOutput(Constants.BIAS);
	}
	public double getDelta() {
		return delta;
	}
	public void setDelta(double delta) {
		this.delta = delta;
	}
	public Double getWeight(int index){
		if(this.weights!=null && !this.weights.isEmpty())
			return this.weights.get(index);
		return null;
	}
	@Override
	public String toString() {
		return "Neuron [weights=" + weights + "]";
	}
	public List<Double> getNewWeights() {
		return newWeights;
	}
	public void setNewWeights(List<Double> newWeights) {
		this.newWeights = newWeights;
	}
	
	
}
