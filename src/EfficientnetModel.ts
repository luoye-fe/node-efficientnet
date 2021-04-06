import * as tf from "@tensorflow/tfjs-node-gpu";
import * as Jimp from "jimp";
import * as sharp from "sharp";
import * as cliProgress from "cli-progress";
import { io } from "@tensorflow/tfjs-core";

import EfficientNetResult from "./EfficientNetResult";

const NUM_OF_CHANNELS = 3;

interface EfficientNetModelInferenceOptions {
  topK?: number;
  locale?: string;
}

export default class EfficientNetModel {
  modelPath: string | io.IOHandler;
  imageSize: number;
  model: tf.GraphModel | undefined;

  constructor(modelPath: string | io.IOHandler, imageSize: number) {
    this.modelPath = modelPath;
    this.imageSize = imageSize;
  }

  async load(): Promise<void> {
    const bar = new cliProgress.SingleBar(
      {},
      cliProgress.Presets.shades_classic
    );
    bar.start(100, 0);
    const model = await tf.loadGraphModel(this.modelPath, {
      onProgress: (p) => {
        bar.update(p * 100);
      },
    });
    bar.stop();
    this.model = model;
  }

  private async createTensor(image: sharp.Sharp): Promise<tf.Tensor3D> {
    const values = new Float32Array(
      this.imageSize * this.imageSize * NUM_OF_CHANNELS
    );

    const metadata = await image.metadata();
    const { width = 0, height = 0 } = metadata;
    const realSize = Math.min(width, height);

    const x = 0;
    const y = 0;
    const w = this.imageSize;
    const h = this.imageSize;

    const bufferData = await image
      .extract({
        left: width > height ? Math.floor((width - height) / 2) : 0,
        top: height > width ? Math.floor((height - width) / 2) : 0,
        width: realSize,
        height: realSize,
      })
      .resize(this.imageSize, this.imageSize)
      .removeAlpha() // keep 3 channels
      .raw()
      .toBuffer({ resolveWithObject: true });

    for (let _y = y; _y < y + h; _y++) {
      for (let _x = x; _x < x + w; _x++) {
        const offset = NUM_OF_CHANNELS * (w * _y + _x);
        values[offset + 0] = ((bufferData.data[offset + 0] - 1) / 127.0) >> 0;
        values[offset + 1] = ((bufferData.data[offset + 1] - 1) / 127.0) >> 0;
        values[offset + 2] = ((bufferData.data[offset + 2] - 1) / 127.0) >> 0;
      }
    }

    const outShape: [number, number, number] = [
      this.imageSize,
      this.imageSize,
      NUM_OF_CHANNELS,
    ];
    let imageTensor = tf.tensor3d(values, outShape, "float32");
    imageTensor = imageTensor.expandDims(0);
    return imageTensor;
  }

  private async predict(
    tensor: tf.Tensor3D,
    topK: number,
    locale: string
  ): Promise<EfficientNetResult> {
    const objectArray = this.model!.predict(tensor) as tf.Tensor;
    const values = objectArray.dataSync() as Float32Array;
    return new EfficientNetResult(values, topK, locale);
  }

  private async createTensorA(image: Jimp): Promise<tf.Tensor3D> {
    const values = new Float32Array(
      this.imageSize * this.imageSize * NUM_OF_CHANNELS
    );
    let i = 0;

    image.scan(
      0,
      0,
      image.bitmap.width,
      image.bitmap.height,
      (x: number, y: number) => {
        const pixel = Jimp.intToRGBA(image.getPixelColor(x, y));
        pixel.r = ((pixel.r - 1) / 127.0) >> 0;
        pixel.g = ((pixel.g - 1) / 127.0) >> 0;
        pixel.b = ((pixel.b - 1) / 127.0) >> 0;
        values[i * NUM_OF_CHANNELS + 0] = pixel.r;
        values[i * NUM_OF_CHANNELS + 1] = pixel.g;
        values[i * NUM_OF_CHANNELS + 2] = pixel.b;
        i++;
      }
    );
    const outShape: [number, number, number] = [
      this.imageSize,
      this.imageSize,
      NUM_OF_CHANNELS,
    ];
    let imageTensor = tf.tensor3d(values, outShape, "float32");
    imageTensor = imageTensor.expandDims(0);
    return imageTensor;
  }

  private async cropAndResize(image: Jimp): Promise<Jimp> {
    const width = image.bitmap.width;
    const height = image.bitmap.height;

    const cropPadding = 32;
    const paddedCenterCropSize =
      ((this.imageSize / (this.imageSize + cropPadding)) *
        Math.min(height, width)) >>
      0;
    const offsetHeight = ((height - paddedCenterCropSize + 1) / 2) >> 0;
    const offsetWidth = (((width - paddedCenterCropSize + 1) / 2) >> 0) + 1;

    await image.crop(
      offsetWidth,
      offsetHeight,
      paddedCenterCropSize,
      paddedCenterCropSize
    );

    await image.resize(this.imageSize, this.imageSize);
    return image;
  }

  async inferenceUseJimp(
    imgPath: string | Buffer,
    options?: EfficientNetModelInferenceOptions
  ): Promise<EfficientNetResult> {
    const { topK = NUM_OF_CHANNELS, locale = "en" } = options || {};
    // @ts-ignore
    let image = await Jimp.read(imgPath);
    image = await this.cropAndResize(image);
    const tensor = await this.createTensorA(image);
    return this.predict(tensor, topK, locale);
  }

  async inference(
    imgPath: string | Buffer,
    options?: EfficientNetModelInferenceOptions
  ): Promise<EfficientNetResult> {
    const { topK = NUM_OF_CHANNELS, locale = "en" } = options || {};
    // @ts-ignore
    const image = sharp(imgPath);
    const tensor = await this.createTensor(image);
    return this.predict(tensor, topK, locale);
  }
}
