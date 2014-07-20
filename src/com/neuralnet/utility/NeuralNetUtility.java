package com.neuralnet.utility;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import com.neuralnet.exec.TrainingSet;


public class NeuralNetUtility {
	
	public static double sigmoid(double activation,double response){
			return 1/(1+Math.pow(Math.E,-1*(activation/response)));
		}
	
	static List<WeightsFileReader> readers ;
	
	
	public static WeightsFileReader getReader(int layerIndex,int neuronIndex){
		if(readers == null)
			return null;
		for(WeightsFileReader reader:readers){
			if(reader.getLayerIndex() == layerIndex && reader.getNeuronIndex()==neuronIndex)
				return reader;
		}
		return null;
	}
	
	public static void readWeightsFile(String weightsFileName){
		readers = new ArrayList<WeightsFileReader>();
		try {
			BufferedReader weightsFileReader = new BufferedReader(new FileReader(weightsFileName));
			String line = null;
			while((line =  weightsFileReader.readLine())!=null && !line.isEmpty() && !line.startsWith(Constants.COMMENT_CHARACTER)){
				String []split = line.split(Constants.SEMICOLON);
				String []weightsString = split[2].split(Constants.SPACE);
				List<Double> weights = new ArrayList<Double>();
				for(String weight : weightsString){
					weights.add(Double.parseDouble(weight));
				}
				WeightsFileReader reader = new WeightsFileReader(Integer.valueOf(split[0]), Integer.valueOf(split[1]), weights );
				readers.add(reader);
			}
			weightsFileReader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void readWeights(String weight){
		if(weight==null || weight.isEmpty())
			return;
		readers = new ArrayList<WeightsFileReader>();
		weight = weight.replace(Constants.CURLY_BRACKET_OPEN, Constants.BLANK);
		weight = weight.replace(Constants.CURLY_BRACKET_CLOSE, Constants.BLANK);
		String []weights = weight.split(Constants.PIPE_CHARACTER);
		for(String line:weights){
			String []split = line.split(Constants.SEMICOLON);
			String []weightsString = split[2].split(Constants.SPACE);
			List<Double> weightsToAdd = new ArrayList<Double>();
			for(String weightSingle : weightsString){
				weightsToAdd.add(Double.parseDouble(weightSingle));
			}
			WeightsFileReader reader = new WeightsFileReader(Integer.valueOf(split[0]), Integer.valueOf(split[1]), weightsToAdd );
			readers.add(reader);
		}
	}
	
	public static List<TrainingSet> getTrainingSet(String inputString) {
		List<TrainingSet> trainingSet =new ArrayList<TrainingSet>();
		inputString = inputString.replace(Constants.CURLY_BRACKET_OPEN, Constants.BLANK);
		inputString = inputString.replace(Constants.CURLY_BRACKET_CLOSE, Constants.BLANK);
		String []inputs = inputString.split(Constants.PIPE_CHARACTER);
		
		for(String input:inputs){
			TrainingSet set = new TrainingSet();
			String []inputToProcess = input.split(Constants.ARROW);
			String []trainingInputs = inputToProcess[0].split(Constants.SEMICOLON);
			String []trainingOutputs = inputToProcess[1].split(Constants.SEMICOLON);
			for(String trainingInput:trainingInputs){
				set.getInputs().add(Double.parseDouble(trainingInput));
			}
			set.getOutputs().add(1.0);
			for(String trainingOutput:trainingOutputs){
				set.getOutputs().add(Double.parseDouble(trainingOutput));
			}
			trainingSet.add(set);
		}
		return trainingSet;
	}
	
}
