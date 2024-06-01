use std::slice;

use zune_image::codecs::png::zune_core::colorspace::{ColorCharacteristics, ColorSpace};

use crate::Img;

// How often to downscale and score the input images.
// Each scaling step will downscale by a factor of two.
const NUM_SCALES: usize = 6;

#[derive(Clone)]
pub struct CpuImg {
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
}

const FROM_SRGB8_TABLE: [f32; 256] = [
    0.0,
    0.000303527,
    0.000607054,
    0.00091058103,
    0.001214108,
    0.001517635,
    0.0018211621,
    0.002124689,
    0.002428216,
    0.002731743,
    0.00303527,
    0.0033465356,
    0.003676507,
    0.004024717,
    0.004391442,
    0.0047769533,
    0.005181517,
    0.0056053917,
    0.0060488326,
    0.006512091,
    0.00699541,
    0.0074990317,
    0.008023192,
    0.008568125,
    0.009134057,
    0.009721218,
    0.010329823,
    0.010960094,
    0.011612245,
    0.012286487,
    0.012983031,
    0.013702081,
    0.014443844,
    0.015208514,
    0.015996292,
    0.016807375,
    0.017641952,
    0.018500218,
    0.019382361,
    0.020288562,
    0.02121901,
    0.022173883,
    0.023153365,
    0.02415763,
    0.025186857,
    0.026241222,
    0.027320892,
    0.028426038,
    0.029556843,
    0.03071345,
    0.03189604,
    0.033104774,
    0.03433981,
    0.035601325,
    0.036889452,
    0.038204376,
    0.039546248,
    0.04091521,
    0.042311423,
    0.043735042,
    0.045186214,
    0.046665095,
    0.048171833,
    0.049706575,
    0.051269468,
    0.052860655,
    0.05448028,
    0.056128494,
    0.057805434,
    0.05951124,
    0.06124607,
    0.06301003,
    0.06480328,
    0.06662595,
    0.06847818,
    0.07036011,
    0.07227186,
    0.07421358,
    0.07618539,
    0.07818743,
    0.08021983,
    0.082282715,
    0.084376216,
    0.086500466,
    0.088655606,
    0.09084173,
    0.09305898,
    0.095307484,
    0.09758736,
    0.09989874,
    0.10224175,
    0.10461649,
    0.10702311,
    0.10946172,
    0.111932434,
    0.11443538,
    0.116970696,
    0.11953845,
    0.12213881,
    0.12477186,
    0.12743773,
    0.13013652,
    0.13286836,
    0.13563336,
    0.13843165,
    0.14126332,
    0.1441285,
    0.1470273,
    0.14995982,
    0.15292618,
    0.1559265,
    0.15896086,
    0.16202943,
    0.16513224,
    0.16826946,
    0.17144115,
    0.17464745,
    0.17788847,
    0.1811643,
    0.18447503,
    0.1878208,
    0.19120172,
    0.19461787,
    0.19806935,
    0.2015563,
    0.20507877,
    0.2086369,
    0.21223079,
    0.21586053,
    0.21952623,
    0.22322798,
    0.22696589,
    0.23074007,
    0.23455065,
    0.23839766,
    0.2422812,
    0.2462014,
    0.25015837,
    0.25415218,
    0.2581829,
    0.26225072,
    0.26635566,
    0.27049786,
    0.27467737,
    0.27889434,
    0.2831488,
    0.2874409,
    0.2917707,
    0.29613832,
    0.30054384,
    0.30498737,
    0.30946895,
    0.31398875,
    0.31854683,
    0.32314324,
    0.32777813,
    0.33245158,
    0.33716366,
    0.34191445,
    0.3467041,
    0.3515327,
    0.35640025,
    0.36130688,
    0.3662527,
    0.37123778,
    0.37626222,
    0.3813261,
    0.38642952,
    0.39157256,
    0.3967553,
    0.40197787,
    0.4072403,
    0.4125427,
    0.41788515,
    0.42326775,
    0.42869055,
    0.4341537,
    0.43965724,
    0.44520125,
    0.45078585,
    0.45641106,
    0.46207705,
    0.46778384,
    0.47353154,
    0.47932023,
    0.48514998,
    0.4910209,
    0.49693304,
    0.5028866,
    0.50888145,
    0.5149178,
    0.5209957,
    0.52711535,
    0.5332766,
    0.5394797,
    0.5457247,
    0.5520116,
    0.5583406,
    0.5647117,
    0.57112503,
    0.57758063,
    0.5840786,
    0.590619,
    0.597202,
    0.60382754,
    0.61049575,
    0.61720675,
    0.62396055,
    0.63075733,
    0.637597,
    0.6444799,
    0.6514058,
    0.65837497,
    0.66538745,
    0.67244333,
    0.6795426,
    0.68668544,
    0.69387203,
    0.70110214,
    0.70837605,
    0.7156938,
    0.72305536,
    0.730461,
    0.7379107,
    0.7454045,
    0.75294244,
    0.76052475,
    0.7681514,
    0.77582246,
    0.78353804,
    0.79129815,
    0.79910296,
    0.8069525,
    0.8148468,
    0.822786,
    0.8307701,
    0.83879924,
    0.84687346,
    0.8549928,
    0.8631574,
    0.87136734,
    0.8796226,
    0.8879232,
    0.89626956,
    0.90466136,
    0.913099,
    0.92158204,
    0.93011117,
    0.9386859,
    0.9473069,
    0.9559735,
    0.9646866,
    0.9734455,
    0.98225087,
    0.9911022,
    1.0,
];

impl CpuImg {
    pub fn from_srgb(src: &[u8], width: usize, height: usize) -> Self {
        let mut data = vec![[0.0, 0.0, 0.0]; width * height];
        assert_eq!(src.len(), width * height * 3);
        for y in 0..height {
            for x in 0..width {
                for i in 0..3 {
                    data[y * width + x][i] =
                        FROM_SRGB8_TABLE[src[y * width * 3 + x * 3 + i] as usize];
                }
            }
        }
        CpuImg {
            data,
            width,
            height,
        }
    }

    pub fn from_planes(planes: &[Vec<f32>; 3], width: usize, height: usize) -> Self {
        let mut data = vec![[0.0, 0.0, 0.0]; width * height];
        // assert_eq!(src.len(), width * height * 3);
        for y in 0..height {
            for x in 0..width {
                for i in 0..3 {
                    data[y * width + x][i] = planes[i][y * width + x];
                }
            }
        }
        CpuImg {
            data,
            width,
            height,
        }
    }

    fn save(&self, name: &str) {
        let samples = unsafe {
            let ptr = self.data.as_ptr();
            let len = self.data.len();
            slice::from_raw_parts(ptr as *const f32, len * 3)
        };
        let mut img =
            zune_image::image::Image::from_f32(samples, self.width, self.height, ColorSpace::RGB);
        img.metadata_mut()
            .set_color_trc(ColorCharacteristics::Linear);
        img.save(format!("./cpu_{name}.png")).unwrap();

        // let mut img = zune_image::image::Image::from_fn(self.width, self.height, ColorSpace::RGB, |y, x, px| {
        //     for i in 0..3 { px[i] = self.data[y * self.width + x][i]; }
        // });
        // img.metadata_mut().set_color_trc(ColorCharacteristics::Linear);
        // img.save(format!("./cpu_{name}.png")).unwrap();
    }
}

/// Computes the SSIMULACRA2 score for a given input frame and the distorted
/// version of that frame.
///
/// # Errors
/// - If the source and distorted image width and height do not match
/// - If the source or distorted image cannot be converted to XYB successfully
/// - If the image is smaller than 8x8 pixels
pub fn compute_frame_ssimulacra2(source: &CpuImg, distorted: &CpuImg) -> f64 {
    let mut img1 = source.clone();
    // img1.save("cpu_src");
    let mut img2 = distorted.clone();

    let mut width = img1.width;
    let mut height = img1.height;

    let mut mul = [
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
        vec![0.0f32; width * height],
    ];
    let mut blur = Blur::new(width, height);
    let mut msssim = Msssim::default();

    for scale in 0..NUM_SCALES {
        if width < 8 || height < 8 {
            break;
        }

        if scale > 0 {
            img1 = downscale_by_2(&img1);
            img2 = downscale_by_2(&img2);
            width = img1.width;
            height = img2.height;
        }
        for c in &mut mul {
            c.truncate(width * height);
        }
        blur.shrink_to(width, height);

        let mut img1 = img1.clone();
        linear_to_xyb(&mut img1);
        let mut img2 = img2.clone();
        linear_to_xyb(&mut img2);
        // img1.save(&format!("ref_xyb_{scale}"));

        // make_positive_xyb(&mut img1);
        // make_positive_xyb(&mut img2);

        // SSIMULACRA2 works with the data in a planar format,
        // so we need to convert to that.
        let img1 = xyb_to_planar(&img1);
        let img2 = xyb_to_planar(&img2);

        image_multiply(&img1, &img1, &mut mul);
        let sigma1_sq = blur.blur(&mul);
        CpuImg::from_planes(&sigma1_sq, width, height).save(&format!("sigma11_{scale}"));

        image_multiply(&img2, &img2, &mut mul);
        let sigma2_sq = blur.blur(&mul);

        image_multiply(&img1, &img2, &mut mul);
        let sigma12 = blur.blur(&mul);

        let mu1 = blur.blur(&img1);
        let mu2 = blur.blur(&img2);

        let avg_ssim = ssim_map(width, height, &mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12);
        let avg_edgediff = edge_diff_map(width, height, &img1, &mu1, &img2, &mu2);
        msssim.scales.push(MsssimScale {
            avg_ssim,
            avg_edgediff,
        });
    }

    msssim.score()
}

fn linear_to_xyb(img: &mut CpuImg) {
    for pix in img.data.iter_mut() {
        let (x, y, b) = px_linear_rgb_to_xyb(pix[0], pix[1], pix[2]);
        pix[0] = x;
        pix[1] = y;
        pix[2] = b;
    }
}

const K_M02: f32 = 0.078f32;
const K_M00: f32 = 0.30f32;
const K_M01: f32 = 1.0f32 - K_M02 - K_M00;

const K_M12: f32 = 0.078f32;
const K_M10: f32 = 0.23f32;
const K_M11: f32 = 1.0f32 - K_M12 - K_M10;

const K_M20: f32 = 0.243_422_69_f32;
const K_M21: f32 = 0.204_767_45_f32;
const K_M22: f32 = 1.0f32 - K_M20 - K_M21;

const K_B0: f32 = 0.003_793_073_4_f32;
const K_B0_ROOT: f32 = 0.1559542025327239180319220163705;
const K_B1: f32 = K_B0;
const K_B2: f32 = K_B0;

const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];
const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [K_B0, K_B1, K_B2];
const OPSIN_ABSORBANCE_BIAS_ROOT: [f32; 3] = [K_B0_ROOT, K_B0_ROOT, K_B0_ROOT];
const INVERSE_OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    11.031_567_f32,
    -9.866_944_f32,
    -0.164_622_99_f32,
    -3.254_147_3_f32,
    4.418_770_3_f32,
    -0.164_622_99_f32,
    -3.658_851_4_f32,
    2.712_923_f32,
    1.945_928_2_f32,
];
const NEG_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [-K_B0, -K_B1, -K_B2];

/// Converts 32-bit floating point linear RGB to XYB. This function does assume
/// that the input is Linear RGB. If you pass it gamma-encoded RGB, the results
/// will be incorrect.
/// Get all components in more or less 0..1 range
pub fn px_linear_rgb_to_xyb(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let (mut rg, mut gr, mut b) = opsin_absorbance(r, g, b);
    rg = rg.max(0.0).cbrt() - OPSIN_ABSORBANCE_BIAS_ROOT[0];
    gr = gr.max(0.0).cbrt() - OPSIN_ABSORBANCE_BIAS_ROOT[1];
    b = b.max(0.0).cbrt() - OPSIN_ABSORBANCE_BIAS_ROOT[2];
    let x = 0.5f32 * (rg - gr);
    let y = 0.5f32 * (rg + gr);
    // Get all components in more or less 0..1 range
    (x.mul_add(14.0, 0.42), y + 0.01, b - y + 0.55)
}

#[inline]
fn opsin_absorbance(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        OPSIN_ABSORBANCE_MATRIX[0].mul_add(
            r,
            OPSIN_ABSORBANCE_MATRIX[1].mul_add(
                g,
                OPSIN_ABSORBANCE_MATRIX[2].mul_add(b, OPSIN_ABSORBANCE_BIAS[0]),
            ),
        ),
        OPSIN_ABSORBANCE_MATRIX[3].mul_add(
            r,
            OPSIN_ABSORBANCE_MATRIX[4].mul_add(
                g,
                OPSIN_ABSORBANCE_MATRIX[5].mul_add(b, OPSIN_ABSORBANCE_BIAS[1]),
            ),
        ),
        OPSIN_ABSORBANCE_MATRIX[6].mul_add(
            r,
            OPSIN_ABSORBANCE_MATRIX[7].mul_add(
                g,
                OPSIN_ABSORBANCE_MATRIX[8].mul_add(b, OPSIN_ABSORBANCE_BIAS[2]),
            ),
        ),
    )
}

// Get all components in more or less 0..1 range
// Range of Rec2020 with these adjustments:
//  X: 0.017223..0.998838
//  Y: 0.010000..0.855303
//  B: 0.048759..0.989551
// Range of sRGB:
//  X: 0.204594..0.813402
//  Y: 0.010000..0.855308
//  B: 0.272295..0.938012
// The maximum pixel-wise difference has to be <= 1 for the ssim formula to make
// sense.
fn make_positive_xyb(xyb: &mut CpuImg) {
    for pix in xyb.data.iter_mut() {
        pix[2] = (pix[2] - pix[1]) + 0.55;
        pix[0] = (pix[0]).mul_add(14.0, 0.42);
        pix[1] += 0.01;
    }
}

fn xyb_to_planar(xyb: &CpuImg) -> [Vec<f32>; 3] {
    let mut out1 = vec![0.0f32; xyb.width * xyb.height];
    let mut out2 = vec![0.0f32; xyb.width * xyb.height];
    let mut out3 = vec![0.0f32; xyb.width * xyb.height];
    for (((i, o1), o2), o3) in xyb
        .data
        .iter()
        .copied()
        .zip(out1.iter_mut())
        .zip(out2.iter_mut())
        .zip(out3.iter_mut())
    {
        *o1 = i[0];
        *o2 = i[1];
        *o3 = i[2];
    }

    [out1, out2, out3]
}

fn image_multiply(img1: &[Vec<f32>; 3], img2: &[Vec<f32>; 3], out: &mut [Vec<f32>; 3]) {
    for ((plane1, plane2), out_plane) in img1.iter().zip(img2.iter()).zip(out.iter_mut()) {
        for ((&p1, &p2), o) in plane1.iter().zip(plane2.iter()).zip(out_plane.iter_mut()) {
            *o = p1 * p2;
        }
    }
}

fn downscale_by_2(in_data: &CpuImg) -> CpuImg {
    const SCALE: usize = 2;
    let in_w = in_data.width;
    let in_h = in_data.height;
    let out_w = (in_w + SCALE - 1) / SCALE;
    let out_h = (in_h + SCALE - 1) / SCALE;
    let mut out_data = vec![[0.0f32; 3]; out_w * out_h];
    let normalize = 1f32 / (SCALE * SCALE) as f32;

    let in_data = &in_data.data;
    for oy in 0..out_h {
        for ox in 0..out_w {
            for c in 0..3 {
                let mut sum = 0f32;
                for iy in 0..SCALE {
                    for ix in 0..SCALE {
                        let x = (ox * SCALE + ix).min(in_w - 1);
                        let y = (oy * SCALE + iy).min(in_h - 1);
                        let in_pix = in_data[y * in_w + x];

                        sum += in_pix[c];
                    }
                }
                let out_pix = &mut out_data[oy * out_w + ox];
                out_pix[c] = sum * normalize;
            }
        }
    }

    CpuImg {
        data: out_data,
        width: out_w,
        height: out_h,
    }
}

fn ssim_map(
    width: usize,
    height: usize,
    m1: &[Vec<f32>; 3],
    m2: &[Vec<f32>; 3],
    s11: &[Vec<f32>; 3],
    s22: &[Vec<f32>; 3],
    s12: &[Vec<f32>; 3],
) -> [f64; 3 * 2] {
    const C2: f32 = 0.0009f32;

    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 2];

    for c in 0..3 {
        let mut sum1 = [0.0f64; 2];
        for (row_m1, (row_m2, (row_s11, (row_s22, row_s12)))) in m1[c].chunks_exact(width).zip(
            m2[c].chunks_exact(width).zip(
                s11[c]
                    .chunks_exact(width)
                    .zip(s22[c].chunks_exact(width).zip(s12[c].chunks_exact(width))),
            ),
        ) {
            for x in 0..width {
                let mu1 = row_m1[x];
                let mu2 = row_m2[x];
                let mu11 = mu1 * mu1;
                let mu22 = mu2 * mu2;
                let mu12 = mu1 * mu2;
                let mu_diff = mu1 - mu2;

                // Correction applied compared to the original SSIM formula, which has:
                //   luma_err = 2 * mu1 * mu2 / (mu1^2 + mu2^2)
                //            = 1 - (mu1 - mu2)^2 / (mu1^2 + mu2^2)
                // The denominator causes error in the darks (low mu1 and mu2) to weigh
                // more than error in the brights (high mu1 and mu2). This would make
                // sense if values correspond to linear luma. However, the actual values
                // are either gamma-compressed luma (which supposedly is already
                // perceptually uniform) or chroma (where weighing green more than red
                // or blue more than yellow does not make any sense at all). So it is
                // better to simply drop this denominator.
                let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
                let num_s = 2f32.mul_add(row_s12[x] - mu12, C2);
                let denom_s = (row_s11[x] - mu11) + (row_s22[x] - mu22) + C2;
                // Use 1 - SSIM' so it becomes an error score instead of a quality
                // index. This makes it make sense to compute an L_4 norm.
                let mut d = 1.0f64 - f64::from((num_m * num_s) / denom_s);
                d = d.max(0.0);
                sum1[0] += d;
                sum1[1] += d.powi(4);
            }
        }
        plane_averages[c * 2] = one_per_pixels * sum1[0];
        plane_averages[c * 2 + 1] = (one_per_pixels * sum1[1]).sqrt().sqrt();
    }

    plane_averages
}

fn edge_diff_map(
    width: usize,
    height: usize,
    img1: &[Vec<f32>; 3],
    mu1: &[Vec<f32>; 3],
    img2: &[Vec<f32>; 3],
    mu2: &[Vec<f32>; 3],
) -> [f64; 3 * 4] {
    let one_per_pixels = 1.0f64 / (width * height) as f64;
    let mut plane_averages = [0f64; 3 * 4];

    for c in 0..3 {
        let mut sum1 = [0.0f64; 4];
        for (row1, (row2, (rowm1, rowm2))) in img1[c].chunks_exact(width).zip(
            img2[c]
                .chunks_exact(width)
                .zip(mu1[c].chunks_exact(width).zip(mu2[c].chunks_exact(width))),
        ) {
            for x in 0..width {
                let d1: f64 = (1.0 + f64::from((row2[x] - rowm2[x]).abs()))
                    / (1.0 + f64::from((row1[x] - rowm1[x]).abs()))
                    - 1.0;

                // d1 > 0: distorted has an edge where original is smooth
                //         (indicating ringing, color banding, blockiness, etc)
                let artifact = d1.max(0.0);
                sum1[0] += artifact;
                sum1[1] += artifact.powi(4);

                // d1 < 0: original has an edge where distorted is smooth
                //         (indicating smoothing, blurring, smearing, etc)
                let detail_lost = (-d1).max(0.0);
                sum1[2] += detail_lost;
                sum1[3] += detail_lost.powi(4);
            }
        }
        plane_averages[c * 4] = one_per_pixels * sum1[0];
        plane_averages[c * 4 + 1] = (one_per_pixels * sum1[1]).sqrt().sqrt();
        plane_averages[c * 4 + 2] = one_per_pixels * sum1[2];
        plane_averages[c * 4 + 3] = (one_per_pixels * sum1[3]).sqrt().sqrt();
    }

    plane_averages
}

#[derive(Debug, Clone, Default)]
struct Msssim {
    pub scales: Vec<MsssimScale>,
}

#[derive(Debug, Clone, Copy, Default)]
struct MsssimScale {
    pub avg_ssim: [f64; 3 * 2],
    pub avg_edgediff: [f64; 3 * 4],
}

impl Msssim {
    // The final score is based on a weighted sum of 108 sub-scores:
    // - for 6 scales (1:1 to 1:32)
    // - for 6 scales (1:1 to 1:32, downsampled in linear RGB)
    // - for 3 components (X + 0.5, Y, B - Y + 1.0)
    // - for 3 components (X, Y, B-Y, rescaled to 0..1 range)
    // - using 2 norms (the 1-norm and the 4-norm)
    // - using 2 norms (the 1-norm and the 4-norm)
    // - over 3 error maps:
    // - over 3 error maps:
    //     - SSIM
    //     - SSIM' (SSIM without the spurious gamma correction term)
    //     - "ringing" (distorted edges where there are no orig edges)
    //     - "ringing" (distorted edges where there are no orig edges)
    //     - "blurring" (orig edges where there are no distorted edges)
    //     - "blurring" (orig edges where there are no distorted edges)
    // The weights were obtained by running Nelder-Mead simplex search,
    // The weights were obtained by running Nelder-Mead simplex search,
    // optimizing to minimize MSE and maximize Kendall and Pearson correlation
    // optimizing to minimize MSE for the CID22 training set and to
    // for training data consisting of 17611 subjective quality scores,
    // maximize Kendall rank correlation (and with a lower weight,
    // validated on separate validation data consisting of 4292 scores.
    // also Pearson correlation) with the CID22 training set and the
    // TID2013, Kadid10k and KonFiG-IQA datasets.
    // Validation was done on the CID22 validation set.
    // Final results after tuning (Kendall | Spearman | Pearson):
    //    CID22:     0.6903 | 0.8805 | 0.8583
    //    TID2013:   0.6590 | 0.8445 | 0.8471
    //    KADID-10k: 0.6175 | 0.8133 | 0.8030
    //    KonFiG(F): 0.7668 | 0.9194 | 0.9136
    #[allow(clippy::too_many_lines)]
    pub fn score(&self) -> f64 {
        const WEIGHT: [f64; 108] = [
            0.0,
            0.000_737_660_670_740_658_6,
            0.0,
            0.0,
            0.000_779_348_168_286_730_9,
            0.0,
            0.0,
            0.000_437_115_573_010_737_9,
            0.0,
            1.104_172_642_665_734_6,
            0.000_662_848_341_292_71,
            0.000_152_316_327_837_187_52,
            0.0,
            0.001_640_643_745_659_975_4,
            0.0,
            1.842_245_552_053_929_8,
            11.441_172_603_757_666,
            0.0,
            0.000_798_910_943_601_516_3,
            0.000_176_816_438_078_653,
            0.0,
            1.878_759_497_954_638_7,
            10.949_069_906_051_42,
            0.0,
            0.000_728_934_699_150_807_2,
            0.967_793_708_062_683_3,
            0.0,
            0.000_140_034_242_854_358_84,
            0.998_176_697_785_496_7,
            0.000_319_497_559_344_350_53,
            0.000_455_099_211_379_206_3,
            0.0,
            0.0,
            0.001_364_876_616_324_339_8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            7.466_890_328_078_848,
            0.0,
            17.445_833_984_131_262,
            0.000_623_560_163_404_146_6,
            0.0,
            0.0,
            6.683_678_146_179_332,
            0.000_377_244_079_796_112_96,
            1.027_889_937_768_264,
            225.205_153_008_492_74,
            0.0,
            0.0,
            19.213_238_186_143_016,
            0.001_140_152_458_661_836_1,
            0.001_237_755_635_509_985,
            176.393_175_984_506_94,
            0.0,
            0.0,
            24.433_009_998_704_76,
            0.285_208_026_121_177_57,
            0.000_448_543_692_383_340_8,
            0.0,
            0.0,
            0.0,
            34.779_063_444_837_72,
            44.835_625_328_877_896,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.000_868_055_657_329_169_8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.000_531_319_187_435_874_7,
            0.0,
            0.000_165_338_141_613_791_12,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.000_417_917_180_325_133_6,
            0.001_729_082_823_472_283_3,
            0.0,
            0.002_082_700_584_663_643_7,
            0.0,
            0.0,
            8.826_982_764_996_862,
            23.192_433_439_989_26,
            0.0,
            95.108_049_881_108_6,
            0.986_397_803_440_068_2,
            0.983_438_279_246_535_3,
            0.001_228_640_504_827_849_3,
            171.266_725_589_730_7,
            0.980_785_887_243_537_9,
            0.0,
            0.0,
            0.0,
            0.000_513_006_458_899_067_9,
            0.0,
            0.000_108_540_578_584_115_37,
        ];

        let mut ssim = 0.0f64;

        let mut i = 0usize;
        for c in 0..3 {
            for scale in &self.scales {
                for n in 0..2 {
                    ssim = WEIGHT[i].mul_add(scale.avg_ssim[c * 2 + n].abs(), ssim);
                    i += 1;
                    ssim = WEIGHT[i].mul_add(scale.avg_edgediff[c * 4 + n].abs(), ssim);
                    i += 1;
                    ssim = WEIGHT[i].mul_add(scale.avg_edgediff[c * 4 + n + 2].abs(), ssim);
                    i += 1;
                }
            }
        }

        ssim *= 0.956_238_261_683_484_4_f64;
        ssim = (6.248_496_625_763_138e-5 * ssim * ssim).mul_add(
            ssim,
            2.326_765_642_916_932f64.mul_add(ssim, -0.020_884_521_182_843_837 * ssim * ssim),
        );

        if ssim > 0.0f64 {
            ssim = ssim
                .powf(0.627_633_646_783_138_7)
                .mul_add(-10.0f64, 100.0f64);
        } else {
            ssim = 100.0f64;
        }

        ssim
    }
}

/// Structure handling image blur.
///
/// This struct contains the necessary buffers and the kernel used for blurring
/// (currently a recursive approximation of the Gaussian filter).
///
/// Note that the width and height of the image passed to [blur][Self::blur] needs to exactly
/// match the width and height of this instance. If you reduce the image size (e.g. via
/// downscaling), [`shrink_to`][Self::shrink_to] can be used to resize the internal buffers.
pub struct Blur {
    kernel: RecursiveGaussian,
    temp: Vec<f32>,
    width: usize,
    height: usize,
}

impl Blur {
    /// Create a new [Blur] for images of the given width and height.
    /// This pre-allocates the necessary buffers.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Blur {
            kernel: RecursiveGaussian,
            temp: vec![0.0f32; width * height],
            width,
            height,
        }
    }

    /// Truncates the internal buffers to fit images of the given width and height.
    ///
    /// This will [truncate][Vec::truncate] the internal buffers
    /// without affecting the allocated memory.
    pub fn shrink_to(&mut self, width: usize, height: usize) {
        self.temp.truncate(width * height);
        self.width = width;
        self.height = height;
    }

    /// Blur the given image.
    pub fn blur(&mut self, img: &[Vec<f32>; 3]) -> [Vec<f32>; 3] {
        [
            self.blur_plane(&img[0]),
            self.blur_plane(&img[1]),
            self.blur_plane(&img[2]),
        ]
    }

    fn blur_plane(&mut self, plane: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; self.width * self.height];
        self.kernel
            .horizontal_pass(plane, &mut self.temp, self.width);
        self.kernel
            .vertical_pass_chunked::<128, 32>(&self.temp, &mut out, self.width, self.height);
        out
    }
}

mod consts {
    #![allow(clippy::unreadable_literal)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

/// Implements "Recursive Implementation of the Gaussian Filter Using Truncated
/// Cosine Functions" by Charalampidis [2016].
pub struct RecursiveGaussian;

impl RecursiveGaussian {
    #[cfg(not(feature = "rayon"))]
    pub fn horizontal_pass(&self, input: &[f32], output: &mut [f32], width: usize) {
        assert_eq!(input.len(), output.len());

        for (input, output) in input
            .chunks_exact(width)
            .zip(output.chunks_exact_mut(width))
        {
            self.horizontal_row(input, output, width);
        }
    }

    fn horizontal_row(&self, input: &[f32], output: &mut [f32], width: usize) {
        let big_n = consts::RADIUS as isize;
        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        let mut n = (-big_n) + 1;
        while n < width as isize {
            let left = n - big_n - 1;
            let right = n + big_n - 1;
            let left_val = if left >= 0 {
                // SAFETY: `left` can never be bigger than `width`
                unsafe { *input.get_unchecked(left as usize) }
            } else {
                0f32
            };
            let right_val = if right < width as isize {
                // SAFETY: this branch ensures that `right` is not bigger than `width`
                unsafe { *input.get_unchecked(right as usize) }
            } else {
                0f32
            };
            let sum = left_val + right_val;

            let mut out_1 = sum * consts::MUL_IN_1;
            let mut out_3 = sum * consts::MUL_IN_3;
            let mut out_5 = sum * consts::MUL_IN_5;

            out_1 = consts::MUL_PREV2_1.mul_add(prev2_1, out_1);
            out_3 = consts::MUL_PREV2_3.mul_add(prev2_3, out_3);
            out_5 = consts::MUL_PREV2_5.mul_add(prev2_5, out_5);
            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;

            out_1 = consts::MUL_PREV_1.mul_add(prev_1, out_1);
            out_3 = consts::MUL_PREV_3.mul_add(prev_3, out_3);
            out_5 = consts::MUL_PREV_5.mul_add(prev_5, out_5);
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if n >= 0 {
                // SAFETY: We know that this chunk of output is of size `width`,
                // which `n` cannot be larger than.
                unsafe {
                    *output.get_unchecked_mut(n as usize) = out_1 + out_3 + out_5;
                }
            }

            n += 1;
        }
    }

    pub fn vertical_pass_chunked<const J: usize, const K: usize>(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert!(J > K);
        assert!(K > 0);

        assert_eq!(input.len(), output.len());

        let mut x = 0;
        while x + J <= width {
            self.vertical_pass::<J>(&input[x..], &mut output[x..], width, height);
            x += J;
        }

        while x + K <= width {
            self.vertical_pass::<K>(&input[x..], &mut output[x..], width, height);
            x += K;
        }

        while x < width {
            self.vertical_pass::<1>(&input[x..], &mut output[x..], width, height);
            x += 1;
        }
    }

    // Apply 1D vertical scan on COLUMNS elements at a time
    pub fn vertical_pass<const COLUMNS: usize>(
        &self,
        input: &[f32],
        output: &mut [f32],
        width: usize,
        height: usize,
    ) {
        assert_eq!(input.len(), output.len());

        let big_n = consts::RADIUS as isize;

        let zeroes = vec![0f32; COLUMNS];
        let mut prev = vec![0f32; 3 * COLUMNS];
        let mut prev2 = vec![0f32; 3 * COLUMNS];
        let mut out = vec![0f32; 3 * COLUMNS];

        let mut n = (-big_n) + 1;
        while n < height as isize {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;
            let top_row = if top >= 0 {
                &input[top as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            let bottom_row = if bottom < height as isize {
                &input[bottom as usize * width..][..COLUMNS]
            } else {
                &zeroes
            };

            for i in 0..COLUMNS {
                let sum = top_row[i] + bottom_row[i];

                let i1 = i;
                let i3 = i1 + COLUMNS;
                let i5 = i3 + COLUMNS;

                let out1 = prev[i1].mul_add(consts::VERT_MUL_PREV_1, prev2[i1]);
                let out3 = prev[i3].mul_add(consts::VERT_MUL_PREV_3, prev2[i3]);
                let out5 = prev[i5].mul_add(consts::VERT_MUL_PREV_5, prev2[i5]);

                let out1 = sum.mul_add(consts::VERT_MUL_IN_1, -out1);
                let out3 = sum.mul_add(consts::VERT_MUL_IN_3, -out3);
                let out5 = sum.mul_add(consts::VERT_MUL_IN_5, -out5);

                out[i1] = out1;
                out[i3] = out3;
                out[i5] = out5;

                if n >= 0 {
                    output[n as usize * width + i] = out1 + out3 + out5;
                }
            }

            prev2.copy_from_slice(&prev);
            prev.copy_from_slice(&out);

            n += 1;
        }
    }
}
