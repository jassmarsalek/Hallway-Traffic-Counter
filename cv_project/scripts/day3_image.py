import cv2

img = cv2.imread('/home/lhsengr09/Hallway-traffic-counter/cv_project/images/test_full.jpg')

print(f"Image shape: {img.shape}")
print(f"Data type: {img.dtype}")

cv2.imshow('My Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
