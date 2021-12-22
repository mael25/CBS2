def project_vehicle(x, y, z, ori_x, ori_y, ori_z):
    pos = np.array([x, y, z])
    ori = np.array([ori_x, ori_y, ori_z])
    ori /= np.linalg.norm(ori)  # Make unit vector

    #new_pos = pos + 4 * ori
    fwd_2d_angle = np.deg2rad(ori_y) #yaw to rad
    new_pos = pos + 5.5 * np.array([np.cos(fwd_2d_angle), np.sin(fwd_2d_angle), 0])
    new_pos_cam_coords = self.converter.convert(np.array([new_pos]))
    if(new_pos_cam_coords.shape[0] == 0):
        return np.array([[192, 147, 0]]) # In the center of the image, almost at the bottom --> stop waypoint
    return new_pos_cam_coords
