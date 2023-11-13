    rec = [None] * 4

    downsample_ratio = 1
    batch_size = 4

    # print("bg_shape", b_g.shape)
    # count = 0
    with torch.no_grad():
        for i, src in enumerate(DataLoader(reader, batch_size=batch_size)):
            if src[0].size(0) == batch_size:

                # torchvision.utils.save_image(src, "src.jpg")
                fgr, pha, *rec = model(src[0].cuda(), *rec, downsample_ratio)
                if i == 0:
                        temp_bg_check = bg
                        try:
                                height, width, ch = reader.shape
                                print("shape", height, width, ch)
                                if height > width:
                                        print("vertical")
                                if 'user_bg_images' in temp_bg_check:
                                        pass
                                elif 'bg_images_v2' in temp_bg_check:
                                        temp_bg_check = temp_bg_check.replace('bg_images_v2', 'bg_images_landscape')

                                elif 'bg_images':
                                        temp_bg_check = temp_bg_check.replace('bg_images', 'bg_images_landscape')

                                else:
#            pass
                                        print("horizontal")
                                bg = cv2.imread(temp_bg_check)
                        except Exception as E:
                                print(E)
                                bg = cv2.imread(bg)
                        print(reader.total_no_frames())
    # rotation_code = reader.rotate_code
                        b_g = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
                        b_g = torch.from_numpy(b_g).permute(2, 0, 1).float().cuda()
                        b_g = b_g[None, :, :, :]

                    new_b, new_c, new_h, new_w = src[1].shape
                    # writer = VideoWriterCV(path='./output_videos/' + video_name, frame_rate=30, w=w, h=h)
                    b_g = F.interpolate(b_g, size=(new_h, new_w),mode='bilinear', align_corners=False)
                    #new_b, new_c, new_h, new_w = src[1].shape
                src[1] = src[1].cuda()
                pha = F.interpolate(pha, size=(new_h, new_w), mode='area')
                com = src[1] * pha*255
                com = (com + b_g*(1-pha))
                writer.write(com)

    writer.close()
    end = time.time()
    print("inference time", end-start)
    update_session(video_name=video_name, status_value="1")
    print(f"THREADS: {len(threading.enumerate())}")
