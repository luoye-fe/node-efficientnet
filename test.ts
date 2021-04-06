import * as fs from "fs";

import {
  EfficientNetCheckPointFactory,
  EfficientNetCheckPoint,
  EfficientNetModel,
} from "./index";

const images = [
  // "car.jpg",
  // "panda.jpg",
  // "gun.jpg",
  "fish.jpg",
  "fish.gif",
  "fish.png",
  "fish.webp",
  "fish.tiff",
  "fish.avif",
  "fish.pdf",
  "fish.heic",
  // "fish.bmp",
  // "car.heic",
  // "panda.heic",
  // "gun.heic",
];
const imageDir = "./tests/examples/";

if (!fs.existsSync(imageDir)) {
  fs.mkdirSync(imageDir);
}

EfficientNetCheckPointFactory.create(EfficientNetCheckPoint.B7, {
  localModelRootDirectory: "./temp",
})
  .then((model: EfficientNetModel) => {
    const logResult: any = {};

    let point = 0;

    images.forEach(async (image) => {
      await Promise.all([
        model.inference(`${imageDir}/${image}`, {
          topK: 3,
          locale: "en",
        }),
        // model.inferenceUseJimp(`${imageDir}/${image}`, {
        //   topK: 3,
        //   locale: "en",
        // }),
      ]).then((result) => {
        logResult[`${point}-----${point}`] = {};

        [1, 1, 1].forEach((i, j) => {
          logResult[`${image}-${j}`] = {
            sharp: `${result[0].result[j].label}(${result[0].result[j].precision})`,
            // jimp: `${result[1].result[j].label}(${result[1].result[j].precision})`,
          };
        });

        point += 1;

        if (point === images.length) {
          console.table(logResult);
        }
      });
    });
  })
  .catch((e: Error) => {
    console.error(e);
  });
