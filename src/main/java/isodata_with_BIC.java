
import ij.IJ;
import ij.ImageJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.plugin.filter.PlugInFilter;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class isodata_with_BIC implements PlugInFilter {

    // image property members
    protected ImagePlus image;
    private String imageTitle;
    private int width;
    private int height;
    private int numPixels;
    private int numBins;
    private float sigma2_hat;
    
    // parameters from dialog
    private int min_k;
    private int max_k;

    @Override
    public int setup(String string, ImagePlus ip) {
        this.image = ip;
        imageTitle = image.getShortTitle();
        if (!showDialog()) {
            return DONE;
        }
        return IJ.setupDialog(image, DOES_8G + DOES_16 + SUPPORTS_MASKING);
    }

    private boolean showDialog() {
        // create dialog window
        GenericDialog gd = new GenericDialog("ISODATA use BIC to find unknown K-value");
        gd.addNumericField("Minimum K value", 2, 0);
        gd.addNumericField("Maximum K value", 10, 0);
        gd.showDialog();
        if (gd.wasCanceled()) {
            return false;
        }
        // get entered values
        min_k = (int) gd.getNextNumber();
        max_k = (int) gd.getNextNumber();

        IJ.log("min. K = " + min_k + "; max. K = " + max_k);
        return true;
    }

    @Override
    public void run(ImageProcessor ip) {
        // get width and height
        width = ip.getWidth();
        height = ip.getHeight();
        numPixels = width * height;
        int depth = image.getBitDepth();
        numBins = (int) Math.round(Math.pow(2, depth));
        IJ.log(imageTitle);
        IJ.log("Width: " + width + ". Height: " + height + ". Total: " + numPixels);
        IJ.log("Greylevel depth = " + depth + ", number of greylevels = " + numBins);

        // stack process
        for (int i = 1; i <= image.getStackSize(); i++) {
            try {
                process(ip);
            } catch (IOException ex) {
                Logger.getLogger(isodata_with_BIC.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
        image.updateAndDraw();
    }

    public void process(ImageProcessor image) throws IOException {
        // load image
        FloatProcessor fp = null;
        fp = image.toFloat(0, fp);
        float[] pixels = (float[]) fp.getPixels();

        // compute histogram
        float[] histo = getHistogram(pixels);

        ArrayList<Float> mu = new ArrayList<Float>();
        float oldBic = Float.NEGATIVE_INFINITY;
        float bic = oldBic;
        float oldSigma2 = 0.0f;
        int k = min_k;
        while (true) {
            oldBic = bic;
            oldSigma2 = sigma2_hat;
            ArrayList<Float> oldMu = new ArrayList<Float>(mu);
            System.out.println("k = " + k);
            // create initial mu
            mu = genRandomMu(k);

            // K mean algorithm
            mu = kMean(histo, mu, k);
            //IJ.log("Singal threshold = " + Integer.toString(getThreshold(mu.get(0), mu.get(1))));

            // Bayesian Information Criterion (BIC)
            bic = getBic(histo, mu, k);
            System.out.println("bic = " + bic);

//            if (oldBic >= bic || k == max_k) {
//                mu = oldMu;
//                sigma2_hat = oldSigma2;
//                break;
//            }
            if (k == max_k) break;
            k++;
        }

        IJ.log("After K-mean training...");
        IJ.log("K = " + k);
        for (float val : mu) {
            IJ.log(Float.toString(val));
        }
        IJ.log("First threshold =" + getThreshold(mu.get(0), mu.get(1)));
        IJ.log("sigma = " + Math.sqrt(sigma2_hat));
    }

    private float[] getHistogram(float[] pixels) {
        float[] histo = new float[numBins];
        Arrays.fill(histo, 0);
        for (int i = 0; i < numPixels; i++) {
            int idx = (int) pixels[i];
            histo[idx]++;
        }
        return histo;
    }

    private float getMean(float[] histo, int lower, int upper) {
        float sumPixVal = 0;
        float numPix = 0;
        for (int i = lower; i <= upper; i++) {
            sumPixVal += i * histo[i];
            numPix += histo[i];
        }
        if (numPix == 0) //handle divide by 0 error
        {
            return 0;
        }

        float mu = sumPixVal / numPix;
        return mu;
    }

    private int getThreshold(float mu0, float mu1) {
        int t = (int) Math.round((mu0 + mu1) / 2.0);
        return t;
    }

    private ArrayList<Float> genRandomMu(int k) {
        ArrayList<Float> mu = new ArrayList<Float>();
        Random randomGen = new Random();
        for (int i = 0; i < k; i++) {
            float ranMu = (float) randomGen.nextInt(numBins - 1);
            mu.add(ranMu);
        }
        Collections.sort(mu);
        return mu;
    }

    private ArrayList<Float> kMean(float[] histo, ArrayList<Float> mu, int k) {
        ArrayList<Float> oldMu = new ArrayList<Float>(mu);
        boolean converge = false;
        int it = 0;
        while (!converge) {
            // compute threshold
            ArrayList<Integer> thresholds = getThresholds(mu, k);

            // compute new mean(mu)
            for (int i = 0; i < (thresholds.size() - 1); i++) {
                float tempMu = getMean(histo, thresholds.get(i) + 1, thresholds.get(i + 1));
                oldMu.set(i, mu.get(i));
                mu.set(i, tempMu);
            }
            for (int i = 0; i < mu.size(); i++) {
                if (Math.round(oldMu.get(i) * 1000) != Math.round(mu.get(i) * 1000)) {
                    converge = false;
                    break;
                } else {
                    converge = true;
                }
            }
            it++;
        }
        System.out.println("After " + it + " iterations.");
        return mu;
    }

    private ArrayList<Integer> getThresholds(ArrayList<Float> mu, int k) {
        ArrayList<Integer> thresholds = new ArrayList<Integer>(k + 1);
        for (int i = 0; i < k; i++) {
            thresholds.add(0);
        }
        thresholds.add(numBins - 1);

        for (int i = 0; i < (k - 1); i++) {
            int t = getThreshold(mu.get(i), mu.get(i + 1));
            thresholds.set(i + 1, t);
        }

        return thresholds;
    }

    private float getBic(float[] histo, ArrayList<Float> mu, int k) {
        // bic(k) = likelihood(k) - penalty(k)
        // likelihood(k) = j(histo, mu, k);
        // penalty(k) = p(k) / 2 * log(numPixels)
        
        float likelihood_k = getLogLikelihood(histo, mu, k);
        //float likelihood_k = getLogLikelihood_method2(histo, mu, k);
        //float likelihood_k = getLogLikelihood_method3(histo, mu, k);
        float penalty_k = getPenalty(k);
        float bic_k = likelihood_k - penalty_k;
        return bic_k;
    }

    private float getLogLikelihood(float[] histo, ArrayList<Float> mu, int k) {
        // j(histo, mu, K) = \sum_{k=1}^{K}(R_k*log(R_k) - R_k*log(R) - 0.5*R_k*log(2*pi) - 0.5*R_k*log(sigma^2) - 0.5*(R_k-K)))
        ArrayList<Integer> thresholds = getThresholds(mu, k);
        float hatSigma2 = getGlobalSigma(histo, mu, thresholds, k);
        sigma2_hat = hatSigma2;
        System.out.println("hatSigma2 = " + hatSigma2);

        double likelihood_k = 0.0;
        for (int i = 0; i < k; i++) {
            int numKPts = getNumberOfPointsInCluster(histo, thresholds, i);
            double p1 = - 0.5 * numKPts * Math.log(Math.PI * 2.0);
            double p2 = - 0.5 * numKPts * Math.log(hatSigma2);
            double p3 = - 0.5 * (double) (numKPts - k);
            double p4 = numKPts * Math.log((double) numKPts);
            double p5 = - numKPts * Math.log((double) numPixels);
            likelihood_k += p1 + p2 + p3 + p4 + p5;
            
//            likelihood_k += (numKPts * Math.log((double) numKPts)
//                    - numKPts * Math.log((double) numPixels)
//                    - 0.5f * numKPts * Math.log(Math.PI * 2.0)
//                    - 0.5f * numKPts * Math.log(hatSigma2)
//                    - 0.5f * (double) (numKPts - k));
            System.out.println("[" + i + "] " + likelihood_k);
        }
        System.out.println("likelihood_k = " + likelihood_k);

        return (float) likelihood_k;
    }
    
    private float getLogLikelihood_method2(float[] histo, ArrayList<Float> mus, int k) {
        // l(D) := log * product(P(x))
        ArrayList<Integer> thresholds = getThresholds(mus, k);
        float hatSigma2 = getGlobalSigma(histo, mus, thresholds, k);
        System.out.println("hatSigma2 = " + hatSigma2);
        sigma2_hat = hatSigma2;
        
        float l_d = 0.0f;   // log-likelihood of the data
        for (int ik = 0; ik < k; ik++) {
            int numKPts = getNumberOfPointsInCluster(histo, thresholds, ik);
            int thresholdLower = thresholds.get(ik);
            int thresholdUpper = thresholds.get(ik + 1);
            double p1 = Math.log(1 / (Math.sqrt(2 * Math.PI * hatSigma2)));
            double rate = (double) numKPts / (double) numPixels;
            double p3 = Math.log(rate);
            for (int i = thresholdLower + 1; i <= thresholdUpper; i++) {
                float muik = mus.get(ik);
                double p2 = (1 / (2 * hatSigma2)) * (i - muik) * (i - muik);
                double temp = p1 - p2 + p3;
                l_d += histo[i] * temp;
            }
        }
        return l_d;
    }
    
    private float getLogLikelihood_method3(float[] histo, ArrayList<Float> mus, int k) {
        // l(D) := log * product(P(x))
        ArrayList<Integer> thresholds = getThresholds(mus, k);
        double hatSigma2 = getGlobalSigma(histo, mus, thresholds, k);
        System.out.println("hatSigma2 = " + hatSigma2);
        sigma2_hat = (float) hatSigma2;
        
        double l_d = 1;   // log-likelihood of the data
        for (int ik = 0; ik < k; ik++) {
            double numKPts = getNumberOfPointsInCluster(histo, thresholds, ik);
            int thresholdLower = thresholds.get(ik);
            int thresholdUpper = thresholds.get(ik + 1);
            for (int i = thresholdLower + 1; i <= thresholdUpper; i++) {
                double muik = mus.get(ik);
                double p1 = numKPts / (double) numPixels;
                double p2 = 1 / Math.sqrt(2 * Math.PI * hatSigma2);
                double x = i;
                double exp = -(1/(2 * hatSigma2)) * (x - muik) * (x - muik);
                double p3 = Math.exp(exp);
                double px = p1 * p2 * p3;
                l_d *= px;
            }
        }
        l_d = Math.log(l_d);
        return (float) l_d;
    }

    private float getGlobalSigma(float[] histo, ArrayList<Float> mu, ArrayList<Integer> thresholds, int k) {
        float sigma2 = 0.0f;
        for (int i = 0; i < k; i++) {
            int thresholdLower = thresholds.get(i);
            int thresholdUpper = thresholds.get(i + 1);
            for (int j = thresholdLower + 1; j <= thresholdUpper; j++) {
                sigma2 += histo[j] * (j - mu.get(i)) * (j - mu.get(i));
            }
        }
        sigma2 = sigma2 / (numPixels - k);
        return sigma2;
    }

    private int getNumberOfPointsInCluster(float[] histo, ArrayList<Integer> thresholds, int clusterNo) {
        int thresholdLower = thresholds.get(clusterNo);
        int thresholdUpper = thresholds.get(clusterNo + 1);
        int sumPoints = 0;
        for (int i = (thresholdLower + 1); i <= thresholdUpper; i++) {
            sumPoints += histo[i];
        }
        return sumPoints;
    }

    private float getPenalty(int k) {
        // penalty(k) = (p(k) / 2) * log(numPixels)
        float pk = 2 * k;
        float log_n = (float) Math.log(numPixels);
        float penalty_k = (pk / 2) * log_n;
        return penalty_k;
    }

    public static void main(String[] args) {
        // set the plugins.dir property to make the plugin appear in the Plugins menu
        Class<?> clazz = isodata_with_BIC.class;
        String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
        String pluginsDir = url.substring(5, url.length() - clazz.getName().length() - 6);
        System.setProperty("plugins.dir", pluginsDir);

        // start ImageJ
        new ImageJ();

        // open the sample
        //ImagePlus image = IJ.openImage("D:\\Images\\TestBlock_P16-9-800-899_cropped_8bit_0800.png");
        ImagePlus image = IJ.openImage("D:\\Images\\P16-9-800-899_cropped_8bit_0800.tif");
        image.show();

        // run the plugin
        IJ.runPlugIn(clazz.getName(), "");
    }
}
