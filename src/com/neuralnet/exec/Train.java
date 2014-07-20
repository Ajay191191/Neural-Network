package com.neuralnet.exec;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import com.neuralnet.NeuralNetwork;
import com.neuralnet.algorithm.Backpropagation;
import com.neuralnet.utility.Constants;
import com.neuralnet.utility.NeuralNetUtility;

public class Train {
	
	private Properties properties ;
	
	
	public static void main(String[] args) {
		Train train = new Train("config/OR_LOGIC");
		train.train();
	}
	
	public Train(String fileName) {
		try {
			this.properties = new Properties();
			properties.load(Train.class.getClassLoader().getResourceAsStream(fileName));
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

	private void train(){
		NeuralNetwork network = new NeuralNetwork(Integer.parseInt(properties.get(Constants.NO_OF_INPUTS)+Constants.BLANK), Integer.parseInt(properties.get(Constants.NO_OF_OUTPUTS)+Constants.BLANK));
		network.createNet(getNodesInLayers(),properties.getProperty(Constants.WEIGHTS));
		List<TrainingSet> trainingSet = NeuralNetUtility.getTrainingSet(properties.getProperty(Constants.TRAINING_SET));
		List<TrainingSet> testingSet = NeuralNetUtility.getTrainingSet(properties.getProperty(Constants.TRAINING_SET));
		
		Backpropagation propagation = new Backpropagation(network);
		propagation.setLearningRate(Double.parseDouble(properties.getProperty(Constants.LEARNING_RATE)));
		propagation.setErrorThreshold(Double.parseDouble(properties.getProperty(Constants.ERROR_THRESHOLD)));
		
		double errorAvg=0;
		int iteration=0;
		while(true){
			errorAvg=0;
			
			for(TrainingSet set:trainingSet){
				network.setInput(set.getInputs());
				propagation.setExpectedOutput(set.getOutputs());
				errorAvg += propagation.train();
			}
			if(errorAvg/trainingSet.size() < Double.parseDouble(properties.getProperty(Constants.AVG_THRESHOLD)) || iteration>Integer.parseInt(properties.getProperty(Constants.LOOPS)))
				break;
			iteration++;
		}
		
		for(TrainingSet set:testingSet){
			network.setInput(set.getInputs());
			network.compute();
			List<Double> outputs = set.getOutputs();
			outputs.remove(0);
			List<Double> output = network.getOutput();
			output.remove(0);
			System.out.println("ExpectedOutput : " + outputs + " Network output " + output);
		}

	}
	
	private List<Integer> getNodesInLayers() {
		List<Integer> layersNodes = new ArrayList<Integer>();
		String layers=properties.getProperty(Constants.NODES_IN_LAYERS);
		String []indLayers = layers.split(Constants.COMMA);
		for(String layer:indLayers){
			layersNodes.add(Integer.parseInt(layer));
		}
		return layersNodes;
	}
	
	public Properties getProperties() {
		return properties;
	}
	public void setProperties(Properties properties) {
		this.properties = properties;
	}

}