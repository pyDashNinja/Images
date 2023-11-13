    batch_size = 4

    with torch.no_grad():
      for i, src in enumerate(DataLoader(reader, batch_size=batch_size)):
        if src[0].size(0) == batch_size:
          fgr, pha, *rec = model(src[0].cuda(), *rec, downsample_ratio)
          if i == 0:
            temp_bg_check = bg
            try:
                print(src[1].shape)
                height, width, ch = src[1].shape
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
                  print("horizontal")


                bg = cv2.imread(temp_bg_check)
            except Exception as E:
                print(E)
                bg = cv2.imread(bg)    # print("inside", b_g.shape)
            b_g = torch.from_numpy(b_g).permute(2, 0, 1).float().cuda()
            b_g = b_g[None, :, :, :]

            new_b, new_c, new_h, new_w = src[1].shape
            b_g = F.interpolate(b_g, size=(new_h, new_w),mode='bilinear', align_corners=False)

          src[1] = src[1].cuda()
          pha = F.interpolate(pha, size=(new_h, new_w), mode='area')
          com = src[1] * pha*255
          com = (com + b_g*(1-pha))

          writer.write(com)

    writer.close()
    end = time.time()

