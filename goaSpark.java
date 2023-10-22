import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.*;
import org.apache.spark.sql.SparkSession;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.*;
import java.util.Map;
import java.lang.Math;

// Author: Naima Noor


public class goaSpark  {

    public static class TupleComparator implements Comparator<Tuple2<String, List<java.lang.Double>>>, Serializable {
        @Override
        public int compare(Tuple2<String, List<java.lang.Double>> x, Tuple2<String, List<java.lang.Double>> y) {
            java.lang.Double fx = fitnessFunc(x._2());
			java.lang.Double fy = fitnessFunc(y._2());
			if (fx > fy) return 1;
			else if (fx < fy) return -1;
			return 0;
        }
    }

	public static double init(double UB, double LB){
		return Math.random()*(UB - LB)+LB;
	}

	public static List<java.lang.Double> listZeros(int dim){
		List<java.lang.Double> zeros = new ArrayList<java.lang.Double>();
		for(int p = 0; p < dim; p++) {
			zeros.add(new java.lang.Double(p));
		}
		return zeros;
	}

	public static double fitnessFunc(List<java.lang.Double> pos){
		double fitness = new java.lang.Double(0);
		for (int p = 0; p < pos.size(); p++){
			fitness += Math.pow(pos.get(p), 2);
		}
		return fitness;
	}

	public static double S_func(double r){
        double f = 0.5;
        double l = 1.5;
        return f*Math.exp(-r/l)-Math.exp(-r); 
    }

	public static double eucledianDistance(List<java.lang.Double> positionA, List<java.lang.Double> positionB){
		double squaredDistance = new java.lang.Double(0);
		for (int p = 0; p < positionA.size(); p++){
			squaredDistance += Math.pow(positionA.get(p) - positionB.get(p), 2);
		}
		return Math.sqrt(squaredDistance);
	}

	public static Tuple2<String, List<java.lang.Double>> rawUpdateNonAggr(Tuple2<String, String> pair, double EPSILON, double UB, double LB, double c){
		// Calculate eq 2.7 without the SUM part
		String[] mainGH = pair._1().split(":");
		String[] neighbourGH = pair._2().split(":");
		List<java.lang.Double> positionGhA = new ArrayList<java.lang.Double>();
		List<java.lang.Double> positionGhB = new ArrayList<java.lang.Double>();
		for (int p = 1; p < mainGH.length; p++){
			positionGhA.add(new java.lang.Double(mainGH[p]));
			positionGhB.add(new java.lang.Double(neighbourGH[p]));
		}
		List<java.lang.Double> S_i = listZeros(positionGhA.size());
		List<java.lang.Double> r_ij_vec = listZeros(positionGhA.size());
		List<java.lang.Double> s_ij = listZeros(positionGhA.size());
		if (!mainGH[0].equals(neighbourGH[0])){
			double dij = eucledianDistance(positionGhA, positionGhB);
			for (int p = 0; p < positionGhA.size(); p++){
				r_ij_vec.set(p, (positionGhA.get(p) - positionGhB.get(p))/(dij + EPSILON));
			}
			double xj_xi = 2 + dij%2;
			for(int p = 0; p < r_ij_vec.size(); p++){
				s_ij.set(p, ((UB - LB) * c / 2)*S_func(xj_xi)*r_ij_vec.get(p));
			}
		}
		Tuple2<String, List<java.lang.Double>> grasshopperPosNonAggr = new Tuple2<>(String.valueOf(mainGH[0]), s_ij);
		return grasshopperPosNonAggr;
	}

	public static Tuple2<String, List<java.lang.Double>> updateGrasshopperPos(Tuple2<String, Iterable<List<java.lang.Double>>> grasshopperPosNonAggr, Tuple2<String, List<java.lang.Double>> tupleTargetPos, double UB, double LB, double c){
		String gsIndex = grasshopperPosNonAggr._1();
		Iterator<List<java.lang.Double>> unAggrs_ij = grasshopperPosNonAggr._2().iterator();
		List<java.lang.Double> S_i = listZeros(grasshopperPosNonAggr._2().iterator().next().size());
		List<java.lang.Double> targetPosition = tupleTargetPos._2();
		while (unAggrs_ij.hasNext()){
			List<java.lang.Double> s_ij = unAggrs_ij.next();
			for (int p = 0; p < s_ij.size(); p++){
				S_i.set(p, S_i.get(p) + s_ij.get(p)); 
			}
		}
		List<java.lang.Double> X_new = listZeros(S_i.size());
		for (int p = 0; p < S_i.size(); p++){
			java.lang.Double pos = c*S_i.get(p) + targetPosition.get(p);
			pos = Math.max(LB, pos);
			pos = Math.min(UB, pos);
			X_new.set(p, pos);
		}
		Tuple2<String, List<java.lang.Double>> newPos = new Tuple2<>(String.valueOf(gsIndex), X_new);
		return newPos;
	}

	public static void main(String[] args) {

		// Initialization
		double cMax = 1;
		double cMin = 0.00004;
		double EPSILON = 1E-14;
		int MAX_GRASSHOPPER = 500;
		int MAX_ITERATION = 10000/MAX_GRASSHOPPER;
		double LB = -1000;
		double UB = 1000;
		int dim = 2;
		double BEST_FITNESS = 0.1;
		double targetFitness  = Integer.MAX_VALUE;
		ArrayList<ArrayList<java.lang.Double>> fitnessPerItr = new ArrayList<ArrayList<java.lang.Double>>();
		
		//	generate random positions and initial target position. 
		List<String> gs = new ArrayList<String>();
		List<java.lang.Double> targetPosition = listZeros(dim);
		for(int i = 0; i < MAX_GRASSHOPPER; i++) {
			List<java.lang.Double> initPosGH = listZeros(dim);
			for (int p = 0; p < dim; p++){
				initPosGH.set(p, init(UB, LB));
			}
			gs.add(String.valueOf(i));
			for (int p = 0; p < dim; p++){
				gs.set(i, gs.get(i) + ":" + String.valueOf(initPosGH.get(p)));
			}
			if (fitnessFunc(initPosGH) < targetFitness){
				targetFitness = fitnessFunc(initPosGH);
				targetPosition = initPosGH;
			}
		}

		System.out.println("================= INITIALIZATION =====================");
		System.out.println("Initial FitnessValue: " + String.valueOf(targetFitness));
		System.out.println("Initial FitnessPosition: ");
		for (double p: targetPosition) System.out.println(p);
		System.out.println("================= INITIALIZATION COMPLETE =====================");


		// Create spark context
		SparkConf conf = new SparkConf().setAppName("GrassHopper Optimization");
		JavaSparkContext jsc = new JavaSparkContext(conf);
		jsc.setLogLevel("ERROR");

		//Initial random grasshoppers
		JavaRDD<String> grasshoppers = jsc.parallelize(gs);

		// Starting the main operation
		long startTime = System.currentTimeMillis();

		// Main iterative loop
		int iter = 1;
		while(iter < MAX_ITERATION && targetFitness > BEST_FITNESS){
			double c = cMax - (iter*(cMax-cMin)/MAX_ITERATION);
			Tuple2<String, List<java.lang.Double>> tupleTargetPos = new Tuple2<>("_", targetPosition);
			JavaPairRDD<String, String> cartesianGS = grasshoppers.cartesian(grasshoppers);
			JavaPairRDD<String, List<java.lang.Double>> nonAggrPos = cartesianGS.mapToPair(pair -> 
																	 rawUpdateNonAggr(pair, EPSILON, UB, LB, c));
			JavaPairRDD<String, List<java.lang.Double>> gsNewPos = nonAggrPos.groupByKey().mapToPair(hopper -> 
			                                                       updateGrasshopperPos(hopper, tupleTargetPos, UB, LB, c));
			Tuple2<String, List<java.lang.Double>> fitnessMin = gsNewPos.min(new TupleComparator());
			grasshoppers = gsNewPos.map(hopper -> {
				String ghPosStr = hopper._1();
				for (java.lang.Double p:hopper._2()) ghPosStr = ghPosStr + ":" + p;
				return ghPosStr;
			});
			if (fitnessFunc(fitnessMin._2()) < targetFitness){
				targetFitness = fitnessFunc(fitnessMin._2());
				targetPosition = fitnessMin._2();
			}

			fitnessPerItr.add(new ArrayList<java.lang.Double>(Arrays.asList(new java.lang.Double(iter), targetFitness)));
			System.out.println("=================================================");
			System.out.println("Completed for iteration: " + String.valueOf(iter));
			System.out.println("Fitness: " +  String.valueOf(targetFitness));
			System.out.println("=================================================");
			iter += 1;
		}
		
		System.out.println("=============== COMPLETED =========================");
		long endtime = System.currentTimeMillis();
		System.out.println( "execution time (sec) = " + (endtime-startTime)/1000);
		fitnessPerItr.forEach(array -> System.out.println(array));
		System.out.println(targetPosition);

	}
}