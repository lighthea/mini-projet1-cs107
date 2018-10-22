package cs107KNN;

class SignatureChecks {

    byte t = 0;
    int extractIntRes = KNN.extractInt(t, t, t, t);
	byte[][][] parseIDXimagesRes = KNN.parseIDXimages(new byte [] {});
	byte[] parseIDXlabelsRes = KNN.parseIDXlabels(new byte [] {});
	float squaredEuclideanDistance = KNN.squaredEuclideanDistance(new byte[][] {}, new byte[][] {});
    float invertedSimilarity = KNN.invertedSimilarity(new byte[][] {}, new byte[][] {});
    int[] quicksortIndicesRes = KNN.quicksortIndices(new float[] {});
    int indexOfMaxRes = KNN.indexOfMax(new int[] {});
    byte electLabelRes = KNN.electLabel(new int[] {}, new byte[] {}, 0);
    byte knnClassifyRes = KNN.knnClassify(new byte[][] {}, new byte[][][] {}, new byte[] {}, 0);

    SignatureChecks() {
        KNN.quicksortIndices(new float[] {}, new int[] {}, 0, 0);
        KNN.swap(0, 0, new float[] {}, new int[] {});
        KMeansClustering.encodeInt(0, new byte[] {}, 0);
        KMeansClustering.recomputeAssignments(new byte[][][] {}, new byte[][][] {}, new int[] {});
        KMeansClustering.recomputeCentroids(new byte[][][] {}, new byte[][][] {}, new int[] {});
    }

    byte[] encodeIDXimagesRes = KMeansClustering.encodeIDXimages(new byte[][][] {});
    byte[] encodeIDXlabelsRes = KMeansClustering.encodeIDXlabels(new byte[] {});
    byte[][][] KMeanseReduceRes = KMeansClustering.KMeansReduce(new byte[][][] {}, 0, 0);
}
