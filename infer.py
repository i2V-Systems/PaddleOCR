import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

VALIDATE_RESULTS = True
SAVE_OUTPUT_FILE = True

if VALIDATE_RESULTS:
    SAVE_WRONG_PREDICTIONS = False

image_dir = "../dataset/wagon/training_data_05-09-2023/test"
rec_model_dir= "./inference/rec_crnn_exp2"
rec_char_dict_path= "../dataset/wagon/training_data_05-09-2023/char_dict.txt"
rec_image_shape = "3,32,100"

if VALIDATE_RESULTS:
    annotations_file = '../dataset/wagon/training_data_05-09-2023/test.txt'
    
    with open(annotations_file, 'r') as txt:
        annotations = txt.read().splitlines()
        
    correct_preds = 0

if __name__ == "__main__":
        
    output = os.popen(
        f'python3 tools/infer/predict_rec.py \
        --image_dir={image_dir} \
        --rec_model_dir={rec_model_dir} \
        --rec_char_dict_path={rec_char_dict_path} \
        --rec_image_shape={rec_image_shape}'
    ).read()
    
    outputs = output.split("\n")

    if SAVE_OUTPUT_FILE:
        f = open("./inference/outputs/infer_outputs.txt", 'w')
    
    total_samples = sum([len(files) for root, dirs, files in os.walk(image_dir)])
    pbar = tqdm(total=total_samples, unit="frame")
    
    for output in outputs:
        if " Predicts of " in output:
            pred_info = output.split(" of ")[-1]
            
            file_path = pred_info.split(":")[0]
            prediction = pred_info.split(":")[1]
        
            prediction_modified = ''
            for pred_char in prediction:
                if pred_char in["(",  ")", "'"]:
                    continue
                prediction_modified += pred_char
                
            pred_text, pred_conf = prediction_modified.split(",")[0], prediction_modified.split(",")[1].strip()
            
            if SAVE_OUTPUT_FILE:
                # f.write(f"{file_path}\t{pred_text}\t{pred_conf}\n")
                f.write(f"{file_path}\t{pred_text}\n")
            
            if VALIDATE_RESULTS:    
                for annotaion in annotations:
                    gt_img_path, gt_annotation = annotaion.split("\t")
                    gt_img_name = gt_img_path.split("/")[-1]
                    
                    if gt_img_name == file_path.split("/")[-1]:
                        if pred_text == gt_annotation:
                            correct_preds += 1
                    
                        elif SAVE_WRONG_PREDICTIONS:
                            img = Image.open(file_path)
                            width, height = img.size
                            new_height = height + 30
                            new_img = Image.new('RGB', (width, new_height), (255, 255, 255))
                            new_img.paste(img, (0, 0))
                            # Draw the text on the new image
                            draw = ImageDraw.Draw(new_img)
                            font = ImageFont.load_default()  # You can change the font if needed
                            text_width, text_height = draw.textsize(pred_text, font=font)
                            # text_position = ((width - text_width) // 2, height + 5)  # Position the text in the center
                            text_position = ((width - text_width) // 2 - 10, height + 5)
                            draw.text(text_position, f"pred: {pred_text}", fill=(255, 0, 0), font=font)
                            
                            draw.text((text_position[0], text_position[1]+10), f"GT  : {gt_annotation}", fill=(0, 0, 0), font=font)
                            
                            output_dir = "./inference/outputs/wrong_predictions"
                            if not os.path.exists(output_dir):
                                os.mkdir(output_dir)

                            # Save the modified image
                            new_img.save(os.path.join(output_dir, file_path.split("/")[-1]))
            
            pbar.update(1)
                    
    pbar.close()
    
    if SAVE_OUTPUT_FILE:
        f.close()
    
    if VALIDATE_RESULTS:
        accuracy = (correct_preds / total_samples) * 100
        print(f"Accuracy  : {accuracy} ({correct_preds}/{total_samples})")