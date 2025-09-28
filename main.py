import cv2
import numpy as np
import math
from scipy import ndimage

def preprocess_image(path, debug=False):
    """
    1. Read the input image
    2. Convert to grayscale
    3. Apply Gaussian blur to reduce noise
    4. Apply thresholding (Adaptive or Otsu)
    5. Use morphological operations: Opening → remove tiny noise, Closing → fill gaps
    """
    # Read the input image
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding - try both Adaptive and Otsu
    # Adaptive threshold
    thresh_adapt = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )
    
    # Otsu threshold
    _, thresh_otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Choose the better threshold based on white pixel ratio
    def white_ratio(im): 
        return (im == 255).sum() / im.size
    
    ratio_adapt = white_ratio(thresh_adapt)
    ratio_otsu = white_ratio(thresh_otsu)
    
    # Choose threshold with reasonable foreground ratio (not too sparse, not too dense)
    if 0.05 <= ratio_adapt <= 0.3:
        thresh = thresh_adapt
        chosen = "adaptive"
    elif 0.05 <= ratio_otsu <= 0.3:
        thresh = thresh_otsu
        chosen = "otsu"
    else:
        # Default to adaptive if both are outside ideal range
        thresh = thresh_adapt
        chosen = "adaptive (default)"
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Opening → remove tiny noise
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing → fill gaps in shapes
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    if debug:
        print(f"preprocess: chosen threshold = {chosen}, white ratio = {white_ratio(thresh):.3f}")
        cv2.imshow("Original", img)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Blurred", gray_blur)
        cv2.imshow("Threshold", thresh)
        cv2.waitKey(1)
    
    return img, gray_blur, thresh

def analyze_edge_curvature(contour):
    """
    Analyze the curvature of a contour to classify it as:
    - circular/dot (high curvature, consistent)
    - straight line (low curvature)
    - curved line (varying curvature)
    """
    if len(contour) < 10:
        return "unknown", 0
    
    # Calculate curvature at multiple points
    curvatures = []
    for i in range(2, len(contour) - 2):
        p1 = contour[i-2][0]
        p2 = contour[i][0]
        p3 = contour[i+2][0]
        
        # Calculate vectors
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        # Calculate angle between vectors
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            curvatures.append(angle)
    
    if not curvatures:
        return "unknown", 0
    
    mean_curvature = np.mean(curvatures)
    curvature_std = np.std(curvatures)
    
    # Classification based on curvature analysis
    if mean_curvature > 0.8 and curvature_std < 0.3:  # High, consistent curvature
        return "circular", mean_curvature
    elif mean_curvature < 0.2:  # Low curvature
        return "straight", mean_curvature
    else:  # Varying curvature
        return "curved", mean_curvature

def detect_shapes_from_edges(binary_img, debug=False):
    """
    Use Canny edges to detect and classify shapes as:
    1. Dots (circular patterns)
    2. Straight lines
    3. Curved lines
    """
    # Apply Canny edge detection
    edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    dots = []
    straight_lines = []
    curved_lines = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20:  # Skip very small contours
            continue
            
        # Analyze the shape based on edge curvature
        shape_type, curvature = analyze_edge_curvature(contour)
        
        if shape_type == "circular":
            # Additional validation for dots
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity > 0.5 and 30 < area < 500:  # Valid dot criteria
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        dots.append((cx, cy, area, circularity))
        
        elif shape_type == "straight":
            # Fit line to contour points
            points = contour.reshape(-1, 2).astype(np.float32)
            if len(points) > 10:
                [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                
                # Calculate line endpoints
                lefty = int((-x * vy / vx) + y) if vx != 0 else int(y)
                righty = int(((binary_img.shape[1] - x) * vy / vx) + y) if vx != 0 else int(y)
                
                line_length = cv2.arcLength(contour, False)
                if line_length > 40:  # Minimum line length
                    straight_lines.append(((0, lefty, binary_img.shape[1]-1, righty), line_length))
        
        elif shape_type == "curved":
            # Store curved line contour
            line_length = cv2.arcLength(contour, False)
            if line_length > 30:  # Minimum curved line length
                curved_lines.append((contour, line_length))
    
    # Filter duplicate dots
    filtered_dots = []
    for dot in dots:
        cx, cy, area, circularity = dot
        is_duplicate = False
        for existing_dot in filtered_dots:
            ex, ey, _, _ = existing_dot
            if np.hypot(cx - ex, cy - ey) < 15:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_dots.append(dot)
    
    # Convert dots to simple (x, y) format
    final_dots = [(x, y) for x, y, _, _ in filtered_dots]
    
    if debug:
        print(f"Shape detection from edges:")
        print(f"  - Total contours: {len(contours)}")
        print(f"  - Circular shapes (dots): {len(final_dots)}")
        print(f"  - Straight lines: {len(straight_lines)}")
        print(f"  - Curved lines: {len(curved_lines)}")
        
        # Visualize detection
        debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Draw dots in red
        for x, y in final_dots:
            cv2.circle(debug_img, (int(x), int(y)), 5, (0, 0, 255), 2)
        
        # Draw straight lines in green
        for line_data in straight_lines:
            line, _ = line_data
            x1, y1, x2, y2 = line
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw curved lines in blue
        for curve_data in curved_lines:
            contour, _ = curve_data
            cv2.drawContours(debug_img, [contour], -1, (255, 0, 0), 2)
        
        cv2.imshow("Edge-based Shape Detection", debug_img)
        cv2.imshow("Canny Edges", edges)
        cv2.waitKey(1)
    
    return final_dots, straight_lines, curved_lines

def detect_dots_accurate(binary_img, debug=False):
    """
    More Accurate Dot Detection for Pulli Kolam:
    1. Use filled contours with significant area (100-800 pixels)
    2. Multiple validation criteria for true dots
    3. Check for circular/elliptical shapes using ellipse fitting
    4. Validate compactness and filled nature
    """
    # Find contours - use RETR_CCOMP to get both outer and inner contours
    contours, hierarchy = cv2.findContours(binary_img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    rejected_reasons = {"area": 0, "circularity": 0, "ellipse_fit": 0, "compactness": 0}
    
    for i, cnt in enumerate(contours):
        # Only consider outer contours (not holes)
        if hierarchy[0][i][3] != -1:  # Skip inner contours
            continue
            
        # Compute area → significant area for true dots [100, 800] pixels
        area = cv2.contourArea(cnt)
        if area < 100 or area > 800:
            rejected_reasons["area"] += 1
            continue
            
        # Compute perimeter
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
            
        # Compute circularity = 4π × area / (perimeter)²
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.6:  # Relaxed circularity but still good
            rejected_reasons["circularity"] += 1
            continue
        
        # Try to fit ellipse - good dots should fit well to ellipse
        if len(cnt) >= 5:  # Need at least 5 points to fit ellipse
            try:
                ellipse = cv2.fitEllipse(cnt)
                # Get ellipse parameters
                (center_x, center_y), (width, height), angle = ellipse
                
                # Check if ellipse is reasonable (not too elongated)
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else float('inf')
                if aspect_ratio > 2.0:  # Too elongated
                    rejected_reasons["ellipse_fit"] += 1
                    continue
                    
                # Check if ellipse area is close to contour area
                ellipse_area = math.pi * (width/2) * (height/2)
                area_ratio = min(area, ellipse_area) / max(area, ellipse_area)
                if area_ratio < 0.7:  # Ellipse doesn't fit well
                    rejected_reasons["ellipse_fit"] += 1
                    continue
                    
            except:
                rejected_reasons["ellipse_fit"] += 1
                continue
        else:
            rejected_reasons["ellipse_fit"] += 1
            continue
        
        # Compactness check - dots should be compact
        bounding_rect = cv2.boundingRect(cnt)
        rect_area = bounding_rect[2] * bounding_rect[3]
        compactness = area / rect_area if rect_area > 0 else 0
        if compactness < 0.5:  # Not compact enough
            rejected_reasons["compactness"] += 1
            continue
            
        # Get centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        dots.append((cx, cy, area, circularity, compactness))
    
    # Filter duplicates and keep best ones
    filtered_dots = []
    min_distance = 20  # Increased minimum distance
    
    for dot in dots:
        x1, y1, area1, circ1, comp1 = dot
        is_duplicate = False
        best_idx = -1
        
        for j, existing_dot in enumerate(filtered_dots):
            x2, y2, area2, circ2, comp2 = existing_dot
            if np.hypot(x1 - x2, y1 - y2) < min_distance:
                # Keep the better dot (higher circularity + compactness)
                score1 = (circ1 + comp1) / 2
                score2 = (circ2 + comp2) / 2
                if score1 > score2:
                    best_idx = j
                is_duplicate = True
                break
                
        if is_duplicate and best_idx >= 0:
            filtered_dots[best_idx] = dot
        elif not is_duplicate:
            filtered_dots.append(dot)
    
    # Convert to simple (x, y, area) format for significant area check
    final_dots = [(x, y, area) for x, y, area, _, _ in filtered_dots]
    
    if debug:
        print(f"detect_dots_accurate: total_contours={len(contours)} -> valid_dots={len(dots)} -> filtered_dots={len(final_dots)}")
        print(f"  - Rejected by area: {rejected_reasons['area']}")
        print(f"  - Rejected by circularity: {rejected_reasons['circularity']}")
        print(f"  - Rejected by ellipse fit: {rejected_reasons['ellipse_fit']}")
        print(f"  - Rejected by compactness: {rejected_reasons['compactness']}")
        
        # Show significant areas
        if final_dots:
            areas = [area for _, _, area in final_dots]
            print(f"  - Dot areas: min={min(areas):.0f}, max={max(areas):.0f}, avg={np.mean(areas):.0f}")
        
        # Show detection results
        debug_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for x, y, area in final_dots:
            radius = int(math.sqrt(area / math.pi))
            cv2.circle(debug_img, (int(x), int(y)), radius, (0, 0, 255), 2)
            cv2.putText(debug_img, f"{int(area)}", (int(x)-20, int(y)-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow("Accurate Dot Detection", debug_img)
        cv2.waitKey(1)
    
    return final_dots



def detect_lines(binary_img, debug=False):
    """
    Improved Line Detection for Padi Kolam - Better detection of significant straight lines:
    1. Apply Canny edge detection with optimized parameters
    2. Use cv2.HoughLinesP with tuned parameters for straight lines
    3. Filter lines by length and merge nearby parallel lines
    4. Group lines by angle (horizontal/vertical dominance common in Padi)
    """
    # Apply Canny edge detection with optimized parameters
    edges = cv2.Canny(binary_img, 30, 100, apertureSize=3)
    
    # Get image dimensions for adaptive thresholds
    h, w = binary_img.shape
    min_line_length = max(60, int(min(h, w) * 0.1))  # At least 10% of image dimension
    
    # Use cv2.HoughLinesP with parameters optimized for straight lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=max(50, int(min(h, w) * 0.05)),  # Adaptive threshold
        minLineLength=min_line_length,
        maxLineGap=max(10, int(min_line_length * 0.2))  # Allow gaps up to 20% of min length
    )
    
    if lines is None:
        if debug:
            print("detect_lines: No lines detected")
            cv2.imshow("Edges", edges)
            cv2.waitKey(1)
        return []
    
    # Enhanced line filtering and grouping
    significant_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        
        # Only keep significant lines (longer than threshold)
        if length < min_line_length:
            continue
            
        # Calculate angle in degrees
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angle = abs(angle)  # Make positive
        if angle > 90:
            angle = 180 - angle  # Normalize to 0-90 degrees
            
        # Prefer horizontal and vertical lines (common in Padi Kolam)
        # Accept lines that are close to horizontal (0°) or vertical (90°)
        is_horizontal = angle <= 15 or angle >= 165  # Within 15° of horizontal
        is_vertical = 75 <= angle <= 105  # Within 15° of vertical
        is_diagonal = 30 <= angle <= 60  # Diagonal lines also common
        
        if is_horizontal or is_vertical or is_diagonal:
            significant_lines.append((line, length, angle))
    
    # Sort by length (longer lines first) and keep the most significant ones
    significant_lines.sort(key=lambda x: x[1], reverse=True)
    
    # Merge nearby parallel lines to avoid counting the same line multiple times
    merged_lines = []
    merge_distance = 20  # pixels
    
    for line_data in significant_lines:
        line, length, angle = line_data
        x1, y1, x2, y2 = line[0]
        
        # Check if this line is too close to an already accepted line
        is_duplicate = False
        for merged_line, _, _ in merged_lines:
            mx1, my1, mx2, my2 = merged_line[0]
            
            # Calculate distance between line midpoints
            mid1_x, mid1_y = (x1 + x2) / 2, (y1 + y2) / 2
            mid2_x, mid2_y = (mx1 + mx2) / 2, (my1 + my2) / 2
            
            if np.hypot(mid1_x - mid2_x, mid1_y - mid2_y) < merge_distance:
                is_duplicate = True
                break
                
        if not is_duplicate:
            merged_lines.append(line_data)
    
    # Extract just the line coordinates
    final_lines = [line for line, _, _ in merged_lines]
    
    if debug:
        print(f"detect_lines: raw_lines={len(lines)} -> significant_lines={len(significant_lines)} -> merged_lines={len(final_lines)}")
        print(f"  - Min line length threshold: {min_line_length}")
        
        # Show edge detection and detected lines
        cv2.imshow("Edges", edges)
        
        # Draw lines on binary image for visualization
        line_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for line in final_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow("Detected Lines", line_img)
        cv2.waitKey(1)
    
    return final_lines

def classify_kolam(dots, lines):
    """
    Improved Classification Rules with better sensitivity to lines:
    - Padi Kolam → Line count >= 3 AND (Dot count <= 5 OR Lines dominate)
    - Pulli Kolam → Dot count >= 8 AND Line count <= 2  
    - Other Kolam → Mixed or unclear patterns
    """
    num_dots = len(dots) if dots else 0
    num_lines = len(lines) if lines else 0
    
    # Padi Kolam → Prioritize line detection (if significant straight lines found)
    if num_lines >= 3 and (num_dots <= 5 or num_lines > num_dots):
        return "Padi Kolam"
    
    # Pulli Kolam → Many dots with few or no lines
    elif num_dots >= 8 and num_lines <= 2:
        return "Pulli Kolam"
    
    # Mixed patterns or unclear
    elif num_dots > 0 and num_lines > 0:
        return "Mixed Kolam (both dots and lines present)"
    
    # Neither dots nor lines detected clearly
    else:
        return "Other/Unclear Pattern"

def draw_features(img, dots, lines):
    """
    Visualization:
    1. Copy original image
    2. Draw detected dots as red circles
    3. Draw detected lines as green lines
    4. Display or save the output image for inspection
    """
    out = img.copy()
    
    # Draw detected dots as red circles
    for x, y in dots:
        cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red filled circles
        cv2.circle(out, (int(x), int(y)), 8, (0, 0, 255), 2)   # Red outline
    
    # Draw detected lines as green lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green lines
    
    return out

def classify_kolam_improved(dots, straight_lines, curved_lines):
    """
    Improved Classification with relative comparison:
    1. Pulli Kolam → Dots with significant area AND dots significantly outnumber lines
    2. Padi Kolam → Both curved and straight lines OR only straight lines dominate
    3. Other Kolam → Only curved lines dominate
    4. Mixed or unclear patterns → Other
    """
    num_dots = len(dots)
    num_straight = len(straight_lines)
    num_curved = len(curved_lines)
    
    # Check for dots with significant area (already filtered in detection)
    significant_dots = 0
    if isinstance(dots, list) and len(dots) > 0:
        if len(dots[0]) >= 3:  # Has area information
            significant_dots = sum(1 for dot in dots if dot[2] >= 150)  # Area >= 150 pixels
        else:
            significant_dots = num_dots  # Assume all detected dots are significant
    
    total_lines = num_straight + num_curved
    total_features = significant_dots + total_lines
    
    if total_features == 0:
        return "No clear Kolam pattern detected"
    
    # Calculate ratios for relative comparison
    dot_ratio = significant_dots / total_features if total_features > 0 else 0
    line_ratio = total_lines / total_features if total_features > 0 else 0
    
    # Classification logic with relative comparison
    
    # 1. Pulli Kolam → Dots with significant area AND dots significantly outnumber lines
    if significant_dots >= 5 and (dot_ratio >= 0.6 or significant_dots > total_lines * 2):
        return f"Pulli Kolam ({significant_dots} significant dots dominate over {total_lines} lines)"
    
    # 2. Padi Kolam → Both curved and straight lines OR only straight lines dominate
    elif (num_straight >= 3 and num_curved >= 2) or (num_straight >= 5 and line_ratio > dot_ratio):
        if num_curved >= 2:
            return f"Padi Kolam (both line types: {num_straight} straight, {num_curved} curved vs {significant_dots} dots)"
        else:
            return f"Padi Kolam (straight lines dominate: {num_straight} straight vs {significant_dots} dots)"
    
    # 3. Other Kolam → Only curved lines dominate
    elif num_curved >= 3 and num_straight <= 1 and line_ratio > dot_ratio:
        return f"Other Kolam (curved lines dominate: {num_curved} curved vs {significant_dots} dots)"
    
    # 4. Check for minimum thresholds with relative comparison
    elif significant_dots >= 3 and significant_dots >= total_lines:
        return f"Pulli Kolam (dots: {significant_dots} >= lines: {total_lines})"
    
    elif total_lines >= 5 and total_lines > significant_dots:
        if num_straight > num_curved:
            return f"Padi Kolam (lines dominate: {total_lines} lines > {significant_dots} dots)"
        else:
            return f"Other Kolam (curved lines dominate: {total_lines} lines > {significant_dots} dots)"
    
    # 5. Mixed or unclear patterns
    else:
        return f"Mixed/Unclear Pattern ({num_straight} straight, {num_curved} curved, {significant_dots} significant dots)"

def main(image_path, debug=False):
    """
    Improved Main pipeline with accurate shape detection:
    1. Preprocessing - Clean binary mask
    2. Accurate Dot Detection - Find dots with significant area
    3. Edge-based Line Detection - Differentiate straight vs curved lines
    4. Improved Classification - Based on your specific requirements
    5. Visualization - Show results with detailed information
    """
    # Step 1: Preprocessing
    img, gray, thresh = preprocess_image(image_path, debug=debug)
    
    # Step 2: Accurate Dot Detection with significant area
    dots = detect_dots_accurate(thresh, debug=debug)
    
    # Step 3: Edge-based Line Detection
    _, straight_lines, curved_lines = detect_shapes_from_edges(thresh, debug=debug)
    
    # Step 4: Improved Classification
    label = classify_kolam_improved(dots, straight_lines, curved_lines)
    
    # Step 5: Results and Analysis
    print(f"Predicted: {label}")
    
    if debug:
        print(f"Detailed Analysis:")
        print(f"  - Dots with significant area: {len(dots)}")
        if dots and len(dots[0]) >= 3:
            areas = [dot[2] for dot in dots]
            significant_count = sum(1 for area in areas if area >= 150)
            print(f"  - Dots with area >= 150: {significant_count}")
            if areas:
                print(f"  - Dot areas range: {min(areas):.0f} - {max(areas):.0f}")
        print(f"  - Straight lines: {len(straight_lines)}")
        print(f"  - Curved lines: {len(curved_lines)}")
    
    # Step 6: Enhanced Visualization
    out = draw_improved_features(img, dots, straight_lines, curved_lines)
    cv2.imshow("Improved Kolam Analysis", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_improved_features(img, dots, straight_lines, curved_lines):
    """
    Improved visualization with area information:
    - Red circles for dots (size based on area)
    - Green lines for straight lines  
    - Blue curves for curved lines
    - Text labels showing areas
    """
    out = img.copy()
    
    # Draw dots with size proportional to area
    for i, dot in enumerate(dots):
        if len(dot) >= 3:  # Has area information
            x, y, area = dot[0], dot[1], dot[2]
            radius = max(5, int(math.sqrt(area / math.pi) / 2))
            cv2.circle(out, (int(x), int(y)), radius, (0, 0, 255), -1)  # Red filled
            cv2.circle(out, (int(x), int(y)), radius + 3, (0, 0, 255), 2)  # Red outline
            # Show area if significant
            if area >= 150:
                cv2.putText(out, f"{int(area)}", (int(x) + 15, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:  # Simple (x, y) format
            x, y = dot[0], dot[1]
            cv2.circle(out, (int(x), int(y)), 8, (0, 0, 255), -1)
    
    # Draw straight lines in green
    for line_data in straight_lines:
        line, _ = line_data
        x1, y1, x2, y2 = line
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green lines
    
    # Draw curved lines in blue
    for curve_data in curved_lines:
        contour, _ = curve_data
        cv2.drawContours(out, [contour], -1, (255, 0, 0), 2)  # Blue curves
    
    # Add legend with counts
    cv2.putText(out, f"Red Dots: {len(dots)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, f"Green Straight: {len(straight_lines)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(out, f"Blue Curved: {len(curved_lines)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return out

def draw_advanced_features(img, dots, straight_lines, curved_lines):
    """
    Enhanced visualization:
    - Red circles for dots
    - Green lines for straight lines  
    - Blue curves for curved lines
    """
    out = img.copy()
    
    # Draw circular dots as red circles
    for x, y in dots:
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)  # Red filled
        cv2.circle(out, (int(x), int(y)), 10, (0, 0, 255), 2)  # Red outline
    
    # Draw straight lines in green
    for line_data in straight_lines:
        line, _ = line_data
        x1, y1, x2, y2 = line
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green lines
    
    # Draw curved lines in blue
    for curve_data in curved_lines:
        contour, _ = curve_data
        cv2.drawContours(out, [contour], -1, (255, 0, 0), 2)  # Blue curves
    
    # Add legend
    cv2.putText(out, "Red: Dots", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, "Green: Straight Lines", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(out, "Blue: Curved Lines", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return out

if __name__ == "__main__":
    # Test the enhanced edge-based detection
    print("Enhanced Kolam Detection with Edge Analysis")
    print("=" * 50)
    
    # Test with one image first with debug output
    main("image_copy.png", debug=True)
    