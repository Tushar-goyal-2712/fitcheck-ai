from django.shortcuts import render
import base64
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import json
import torchvision
from ultralytics import YOLO

def camera_view(request):
    return render(request, 'camera.html')

# This function is for handling both uploaded and captured images
@csrf_exempt
def upload_photo(request):
    if request.method == 'POST':
        try:
            # Check if it's a base64 image data (for captured images)
            if request.content_type == 'application/json':
                body = json.loads(request.body.decode('utf-8'))
                image_data = body.get('image', None)

                if not image_data:
                    return JsonResponse({'error': 'No image data provided'}, status=400)

                # Decode Base64 Image
                image_base64 = image_data.split(',')[1]
                image_binary = base64.b64decode(image_base64)

                # Save the Image to Static Directory
                static_dir = os.path.join(settings.STATICFILES_DIRS[0])
                image_path = os.path.join(static_dir, 'image.jpg')

                with open(image_path, 'wb') as f:
                    f.write(image_binary)

            # Check if it's a file upload (for uploaded images from gallery)
            elif request.FILES.get('file'):
                uploaded_file = request.FILES['file']
                static_dir = os.path.join(settings.STATICFILES_DIRS[0])
                image_path = os.path.join(static_dir, 'image.jpg')

                # Save the uploaded image
                with open(image_path, 'wb') as f:
                    for chunk in uploaded_file.chunks():
                        f.write(chunk)

            else:
                return JsonResponse({'error': 'No valid image data or file provided'}, status=400)

            # Initialize the YOLO model
            model = YOLO(os.path.join(settings.DETECTION_MODEL_DIR, 'best.pt'))
            
            # Run the model on the image
            results = model([image_path])
            person_type = []
            class_names = model.names
            fat_person_detected = False  # Flag to check if fat_person is detected
            fat_person_confidence_low = False  # Flag to check if fat_person's confidence is low
            normal_detected = False  # Flag to check if normal_person is detected

            for result in results:
                boxes = result.boxes
                probs = result.boxes.conf

                if len(boxes) > 0:
                    keep_indices = torchvision.ops.nms(boxes.xyxy[:, :4], probs, iou_threshold=0.3)
                    result.boxes = result.boxes[keep_indices]

                for idx, box in enumerate(result.boxes):
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    class_name = class_names[class_id] if class_id in class_names else "Unknown"
                    person_type.append((class_name, confidence))
                    
            # Check for fat_persone class and confidence threshold
            if not person_type:
                response_message = "No individuals detected. Please ensure the image is clear and contains visible people."
                return JsonResponse({'message': response_message})
            
            if len(person_type)> 1:
                response_message = "Multiple individuals detected—please try capturing one person at a time!"
                return JsonResponse({'message': response_message})
            
            if len(person_type) == 1 and person_type[0][0] == "fat_persone":
                confidence_value = person_type[0][1].item()  # Get the numerical value of the tensor
                if confidence_value < 0.4:
                    fat_person_confidence_low = True
                else:
                    fat_person_detected = True
            
            elif len(person_type) == 1 and person_type[0][0] == "normal":
                normal_detected = True
            
            else:
                response_message = "There is some error! Please try again later"
                return JsonResponse({'message': response_message})

            # Remove the temporary image file
            os.remove(image_path)

            # Generate response message based on the detections
            if fat_person_detected:
                response_message = "Signs of higher body weight detected. Small changes can lead to big improvements!"
            elif fat_person_confidence_low:
                response_message = "Great news! You seem to be on the right track. Keep going!"
            elif normal_detected:
                response_message = "Great news! You seem to be on the right track. Keep going!"
            else:
                response_message = "There is some error! Please try again later"

            # Return the detection result message
            return JsonResponse({'message': response_message})

        except Exception as e:
            return JsonResponse({'error': f'Failed to upload image: {str(e)}'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)
