package cs107KNN;

import java.util.Arrays;



public class KNN {

    public static void main(String[] args) {

        int columns = 10, lines = 10;
        int test = columns * lines;
        int K = 10;

        byte[][][] trainImages = parseIDXimages(Helpers.readBinaryFile("datasets/Train/5000-per-digit_images_train"));
        byte[] trainLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/Train/5000-per-digit_labels_train"));
        byte[][][] testImages = parseIDXimages(Helpers.readBinaryFile("datasets/Test/10k_images_test"));
        byte[] testLabels = parseIDXlabels(Helpers.readBinaryFile("datasets/Test/10k_labels_test"));
        byte[] predictions = new byte[test];

        long start = System.currentTimeMillis();
        assert testImages != null;
        assert trainLabels != null;

        for(int i =0; i< test; ++i) {

            predictions[i] = knnClassify(testImages[i],trainImages, trainLabels, K);
        }

        double time = (System.currentTimeMillis() - start)/1000d;

        System.out.println("Accuracy : " + accuracy(predictions, Arrays.copyOfRange(testLabels, 0, test)));
        System.out.println("Time : " + time + " seconds !");
        System.out.println("Time per image : " + time/ test + " seconds !");
        Helpers.show("test", testImages, predictions, testLabels, lines, columns);

    }

    public static int extractInt(byte b31ToB24, byte b23ToB16, byte b15ToB8, byte b7ToB0) {

	    /* Combine the four bytes with the OR operator in order to create a virtual 32 bit long byte (using OR the value of
		   the bits is conserved).

		   Apply a 0xFF mask to ensure there will be no problem with the sign

		   Combining requires shifting to place the bits in the right place so the bytes are shifted respecting
		   their first index value

		 */

        return b7ToB0 & 0xFF | (b15ToB8 & 0xFF) << 8 | (b23ToB16 & 0xFF) << 16 | (b31ToB24 & 0xFF) << 24;
    }

    public static float squaredEuclideanDistance( byte[][] a, byte[][] b) {
        float sum = 0;

        //We should only compare similar matrices
        assert (a.length == b.length && a.length != 0) && (a[0].length == b[0].length && a[0].length != 0);


        int lenI = a.length, lenJ = a[0].length;

        //Proper calculation of the squared-euclidean distance
        for (int i = 0; i < lenI; i++) {
            for (int j = 0; j < lenJ; j++) {
                sum += Math.pow(a[i][j] - b[i][j], 2);
            }
        }

        return sum;
        }
    /**
     * Parses an IDX file containing images into a computer-readable array
     *
     * @param data the binary content of the file
     *
     * @return A tensor of images
     */
    public static byte[][][] parseIDXimages( byte[] data) {

        //verifying the magic number
        if (extractInt(data[0], data[1], data[2], data[3]) != 2051)  return null;

        //reading metadata
        int nImages     = extractInt(data[4], data[5], data[6], data[7]);
        int imageHeight = extractInt(data[8], data[9], data[10], data[11]);
        int imageWidth  = extractInt(data[12], data[13], data[14], data[15]);

        for (int i = 0; i < nImages * imageHeight * imageWidth; i++) data[i] = (byte) ((data[i] & 0xFF) - 128);
        byte[][][] tensor = new byte[nImages][imageHeight][imageWidth];
        for (int pixel = 16, row = 0, image = 0; (image < nImages); pixel += imageWidth,
                row = (row + 1) % imageHeight, image += (row + 1)/ imageHeight) {
            //Using integer division to avoid branching in critical loop for image number

            //Inserting the right pixel in the right place with array copy to avoid branching for rows and columns
            tensor[image][row] =  Arrays.copyOfRange(data, pixel, (pixel + imageHeight));

        }
        return tensor;
    }

    /**
     * Parses an IDX image containing labels into a computer-readable array
     *
     * @param data the binary content of the file
     *
     * @return the parsed labels
     */
    public static byte[] parseIDXlabels(byte[] data) {
        //verifying the magic number
        if (extractInt(data[0], data[1], data[2], data[3]) != 2049) { return null; }

        //reading metadata
        int nLabels   = extractInt(data[4], data[5], data[6], data[7]);
        //outputs the rest of the data
        return Arrays.copyOfRange(data, 8, nLabels + 8);
    }
    /**
     * @brief Computes the average pixel value of an image
     *
     * @param a an image
     *
     * @return the average pixel value of the image
     */
    public static float average (byte[][] a) {

        float average = 0;
        int lenI = a.length, lenJ = a[0].length;

        for (int i = 0; i < lenI; ++i)
            for (int j = 0; j < lenJ; ++j)
                average += a[i][j];

        return average / (lenI * lenJ);
        }
    /**
     * @brief Computes the inverted similarity between two images.
     *
     * @param a, b two images of the same size
     *
     * @return the inverted similarity between the two images
     */

    public static float invertedSimilarity(byte[][] a, byte[][] b) {
        assert (a.length == b.length && a.length != 0) && (a[0].length == b[0].length && a[0].length != 0);

        float denominator;
        float numerator = 0;

        float moyenneA  = average(a);
        float moyenneB  = average(b);

        float moyenneAminus;
        float moyenneBminus;


        /**************************************************/
        float temp1 =0; //Première partie du dénominateur
        float temp2 =0; //Seconde partie du dénominateur

        for(int i = 0; i < a.length; ++i) {
            for(int j = 0; j < a[0].length; ++j) {
                moyenneAminus = a[i][j] - moyenneA;
                moyenneBminus = b[i][j] - moyenneB;

                numerator   += (moyenneAminus * moyenneBminus); // Aij - Average(A) * Bij - Average(B)
                temp2       += Math.pow(moyenneBminus,2);// (Bij - Average(B))^2
                temp1       += Math.pow(moyenneAminus,2);// (Aij - Average(A))^2
            }
        }
        denominator = (float) Math.sqrt(temp1 * temp2);
        /***************************************************/

        if(denominator == 0 ) {return 2;}
        else {return (float) 1 - (numerator/denominator) ;}
}
    /**
     * @brief Sorts the array using the Quicksort algorithm
     *
     * @param values, the array to sort
     *
     * @return the indices of the input array sorted so that they correspond to the sorted version of the input array
     */
    public static int[] quicksortIndices(float[] values) {
        int[] indices = new int[values.length];

        for(int i = 0; i < values.length; ++i) {
            indices[i] = i;
        }

        int low  = 0;
        int high = ( values.length ) - 1;

        quicksortIndices( values, indices, low, high );

        return indices;
    }

    public static void quicksortIndices(float[] values, int[] indices, int low, int high) {
        int l       = low;
        int h       = high;
        float pivot = values[low];
        while( l <= h ) {
            if( values[l] < pivot ) {
                ++l;
            } else if(values[h] > pivot) {
                --h;
            } else {
                swap(l, h, values, indices);
                ++l;
                --h;
            }
        }
        if(low < h) {
            quicksortIndices(values, indices, low, h);
        }
        if(high > l) {
            quicksortIndices(values, indices, l, high);
        }
    }
    /**
     * @brief Sorts the array using the Quicksort algorithm
     *
     * @param values, the array to sort
     *
     * @return the indices of the input array sorted so that they correspond to the sorted version of the input array
     */
    public static void swap(int i, int j, float[] values, int[] indices) {
        float tmpFloat;
        int tmpInt;
        tmpFloat  = values[i];
        values[i] = values[j];
        values[j] = tmpFloat;

        tmpInt      = indices[i];
        indices[i]  = indices[j];
        indices[j]  = tmpInt;

    }
    /**
     * @brief Finds the index if the highest value in the array
     *
     * @param array, the array in which the element has to be found
     *
     * @return the highest value indices of the array
     */
    public static int indexOfMax(int[] array) {
        float[] floats = new float[array.length];

        for(int i = 0; i < array.length; ++i) {
            floats[i] = array[i];
        }

        int[] indices = quicksortIndices(floats);

        return indices[array.length - 1];
    }
    /**
     * @brief Chooses the label of an image, considering its closest neighbours
     *
     * @param sortedIndices, labels, k : the array of sorted indices to use, the labels to use, the number of K-nearest neighbours to consider
     *
     * @return the label chosen for the image group
     */
    public static byte electLabel(int[] sortedIndices, byte[] labels, int k) {
        assert k <= sortedIndices.length;
        int results[] = new int[10];

        for(int i =0; i<k; ++i) {
            ++results[labels[sortedIndices[i]]];
        }

        return (byte) (indexOfMax(results));
    }

    public static void swap(int i, int j, double[] values, int[] indices) {
        double tmpFloat;
        int tmpInt;

        tmpFloat    = values[i];
        values[i]   = values[j];
        values[j]   = tmpFloat;

        tmpInt      = indices[i];
        indices[i]  = indices[j];
        indices[j]  = tmpInt;
    }

    public static void quicksortIndices(double[] values, int[] indices, int low, int high) {
        int l   = low;
        int h   = high;
        double pivot = values[low];

        while(l <= h) {
            if(values[l] < pivot) {
                ++l;
            } else if(values[h] > pivot) {
                --h;
            } else {
                swap(l, h, values, indices);
                ++l;
                --h;
            }
        }
        if(low < h) {
            quicksortIndices(values, indices, low, h);
        }
        if(high > l) {
            quicksortIndices(values, indices, l, high);
        }

    }

    public static int[] quicksortIndices(double[] values) {
        assert values.length != 0;

        int len         = values.length;
        int[] indices   = new int[len];

        for(int i = 0; i < len; ++i) {
            indices[i] = i;
        }

        int low     = 0;
        int high    = len-1;

        quicksortIndices(values, indices, low, high);

        return indices;
    }
    /**
     * @brief Main function to use to classify the images
     *
     * @param image, trainImages, trainLabels, k, the image to elect, the images to train on, the labels of rhe images to train
     *
     * @return the label of an image
     */
    public static byte knnClassify(byte[][] image, byte[][][] trainImages, byte[] trainLabels, int k) {
        assert image != null;

        double [] similarity = new double[trainLabels.length];

        Arrays.parallelSetAll(similarity, i -> invertedSimilarity(image, trainImages[i]));
        assert similarity.length != 0;
        int[] sortedIndices = quicksortIndices(similarity);

        return electLabel(sortedIndices, trainLabels, k);
    }
    /**
     * @brief mesure the accuracy of the predictions
     *
     * @param predictedLabels, trueLabels, the label predicted and the true label
     *
     * @return a percentage of accuracy
     */
    public static double accuracy(byte[] predictedLabels, byte[] trueLabels) {
        int counter = 0;
        for(int i = 0; i < trueLabels.length; ++i) {
            if(predictedLabels[i]==trueLabels[i]) {
                ++counter;
            }
        }
        return (double)counter / (double)trueLabels.length;
    }
}