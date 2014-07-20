package com.neuralnet.exec;

import java.util.ArrayList;
import java.util.List;

public class TrainingSet {
	private List<Double> inputs;
	private List<Double> outputs;
	public List<Double> getInputs() {
		return inputs;
	}
	public void setInputs(List<Double> inputs) {
		this.inputs = inputs;
	}
	public List<Double> getOutputs() {
		return outputs;
	}
	public void setOutputs(List<Double> outputs) {
		this.outputs = outputs;
	}
	public TrainingSet(List<Double> inputs, List<Double> outputs) {
		super();
		this.inputs = inputs;
		this.outputs = outputs;
	}
	public TrainingSet() {
		this.inputs = new ArrayList<Double>();
		this.outputs = new ArrayList<Double>();
	}
}
