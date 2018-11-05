package cs107KNN;
import java.util.*;


public class KMeansClustering {
	public static void main(String[] args) {
		int K = 1000;
		int maxIters = 20;

		byte[][][] images = KNN.parseIDXimages(Helpers.readBinaryFile("datasets/Test/10k_images_test"));
		byte[] labels = KNN.parseIDXlabels(Helpers.readBinaryFile("datasets/Test/10k_labels_test"));

		byte[][][] reducedImages = KMeansReduce(images, K, maxIters);

		byte[] reducedLabels
                = new byte[reducedImages.length];
		for (int i = 0; i < reducedLabels.length; i++) {
			reducedLabels[i] = KNN.knnClassify(reducedImages[i], images, labels, 5);
			System.out.println("Classified " + (i + 1) + " / " + reducedImages.length);
		}

		Helpers.show("Titre", reducedImages, reducedLabels, 10, 10);
		Helpers.writeBinaryFile("datasets/reduced10Kto1K_images", encodeIDXimages(reducedImages));
		Helpers.writeBinaryFile("datasets/reduced10Kto1K_labels", encodeIDXlabels(reducedLabels));

	}


    /**
     * @brief Encodes a tensor of images into an array of data ready to be written on a file
     * 
     * @param images the tensor of image to encode
     * 
     * @return the array of byte ready to be written to an IDX file
     */
	public static byte[] encodeIDXimages(byte[][][] images) {
        int depth = images.length,imageWidth = images[0][0].length, imageHeight = images[0].length;

	    byte [] imageIDX = new byte[16 + depth * imageHeight * imageWidth];
        encodeInt(2051, imageIDX, 0);
        encodeInt(images.length, imageIDX, 4);
        encodeInt(images[0].length, imageIDX, 8);
        encodeInt(images[0][0].length, imageIDX, 12);

        for (int i = 0; i < depth ; i++)
            for (int j = 0; j < imageHeight; j++) {
                for (int k = 0; k < imageWidth; k++) {
                    images[i][j][k] = (byte) ((images[i][j][k] & 0xFF) + 128);
                }
            }

		for (int row = 0, image = 0, pixel = 16;
             image < depth;
             pixel += imageWidth, image += (row+1)/imageHeight, row = (row+1)%imageHeight) {

		    System.arraycopy(images[image][row], 0, imageIDX, pixel, imageWidth);
		}
        return imageIDX;

	}

    /**
     * @brief Prepares the array of labels to be written on a binary file
     * 
     * @param labels the array of labels to encode
     * 
     * @return the array of bytes ready to be written to an IDX file
     */
	public static byte[] encodeIDXlabels(byte[] labels) {
		byte [] labelIDX = new byte[8 + labels.length];
	    encodeInt(2049, labelIDX, 0);
        encodeInt(labels.length, labelIDX, 4);
        System.arraycopy(labels, 0, labelIDX, 8, labels.length);

		return labelIDX;
	}

    /**
     * @brief Decomposes an integer into 4 bytes stored consecutively in the destination
     * array starting at position offset
     * @param n the integer number to encode
     * @param destination the array where to write the encoded int
     * @param offset the position where to store the most significant byte of the integer,
     * the others will follow at offset + 1, offset + 2, offset + 3
     */
	public static void encodeInt(int n, byte[] destination, int offset) {

	    destination[offset] = (byte)(n >>> 24);
	    destination[offset+1] = (byte)(n >>> 16);
	    destination[offset+2] = (byte)(n >>> 8);
	    destination[offset+3] = (byte)(n);
	}

    /**
     * @brief Runs the KMeans algorithm on the provided tensor to return size elements.
     * 
     * @param tensor the tensor of images to reduce
     * @param size the number of images in the reduced dataset
     * @param maxIters the number of iterations of the KMeans algorithm to perform
     * 
     * @return the tensor containing the reduced dataset
     */
    public static byte[][][] KMeansReduce(byte[][][] tensor, int size, int maxIters) {
        int[] assignments = new Random().ints(tensor.length, 0, size).toArray();
        byte[][][] centroids = new byte[size][][];
        initialize(tensor, assignments, centroids);

        int nIter = 0;


        while (nIter < maxIters) {
            // Step 1: Assign points to closest centroid
            recomputeAssignments(tensor, centroids, assignments);
            System.out.println("Recomputed assignments");
            // Step 2: Recompute centroids as average of points
            recomputeCentroids(tensor, centroids, assignments);
            System.out.println("Recomputed centroids");

            System.out.println("KMeans completed iteration " + (nIter + 1) + " / " + maxIters);

            nIter++;
        }

        return centroids;
    }

   /**
     * @brief Assigns each image to the cluster whose centroid is the closest.
     * It modifies.
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */
	public static void recomputeAssignments(byte[][][] tensor, byte[][][] centroids, int[] assignments) {
	    assert assignments.length == tensor.length;
        float temp, temp2;

        for (int i = 0; i < assignments.length; i++) {
            for (int j = 0; j < centroids.length; j++) {
                temp = KNN.squaredEuclideanDistance(tensor[i], centroids[j]);
                temp2 = KNN.squaredEuclideanDistance(tensor[i], centroids[assignments[i]]);
                if (temp < temp2){
                    assignments[i] = j;
                }
            }
        }
	}

    /**
     * @brief Computes the centroid of each cluster by averaging the images in the cluster
     * 
     * @param tensor the tensor of images to cluster
     * @param centroids the tensor of centroids that represent the cluster of images
     * @param assignments the vector indicating to what cluster each image belongs to.
     *  if j is at position i, then image i belongs to cluster j
     */
    public static void recomputeCentroids(byte[][][] tensor/*images*/, byte[][][] centroids/*imagesmoyennes*/, int[] assignments) {
        centroids = new byte[centroids.length][centroids[0].length][centroids[0][0].length];
        float compteur = 0;

        for (int i = 0; i < assignments.length; i++)/*pour chaque image d'indice i*/ {
            for (int u = 0; u < centroids[0].length; u++)/*pour chaque ligne*/ {
                for (int v = 0; v < centroids[0][0].length; v++)/*pour chaque colonne*/  {
                    compteur += 1;
                    centroids[assignments[i]][u][v] += (1 / (compteur) * (tensor[assignments[i]][u][v] & 0xFF));
                }
            }
        compteur = 0;
        }
    }
    /**
     * Initializes the centroids and assignments for the algorithm.
     * The assignments are initialized randomly and the centroids
     * are initialized by randomly choosing images in the tensor.
     * 
     * @param tensor the tensor of images to cluster
     * @param assignments the vector indicating to what cluster each image belongs to.
     * @param centroids the tensor of centroids that represent the cluster of images
     *  if j is at position i, then image i belongs to cluster j
     */
    public static void initialize(byte[][][] tensor, int[] assignments, byte[][][] centroids) {
        Set<Integer> centroidIds = new HashSet<>();
        Random r = new Random("cs107-2018".hashCode());
        while (centroidIds.size() != centroids.length)
            centroidIds.add(r.nextInt(tensor.length));
        Integer[] cids = centroidIds.toArray(new Integer[] {});
        for (int i = 0; i < centroids.length; i++)
            centroids[i] = tensor[cids[i]];
        for (int i = 0; i < assignments.length; i++)
            assignments[i] = new Random().nextInt(centroids.length);
    }
}
