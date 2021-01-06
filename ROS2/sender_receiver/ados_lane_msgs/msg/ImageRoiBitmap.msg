std_msgs/Header          header
#
# Original Image Size Resolution (e.g. w=1280 h=720)
#

uint32 orig_width       # image width, that is, number of columns of the original image as provided by cam driver
uint32 orig_height      # image height, that is, number of rows of the original image as provide by cam driver

#
# The ROI is defined by a rectangle.
# We can also look at this as two ranges, one for the x values and one for the y values.
# The image coordinate system has origin in the top left corner with
# x increasing to the right and
# y increasing to the bottom.
#

uint32 roi_x        # x coordinate of the left of the roi (smallest value of x). 0 if uncropped
uint32 roi_y        # y coordinate of the top of the roi (smallest value of y). 0 if uncropped
uint32 roi_width    # width of the roi. orig_width if uncropped
uint32 roi_height   # height of roi. orig_height if uncropped

#
# A bitmap is generated (e.g. by CNN's) which may have resolution different from that of the ROI
# This is the size of the bitmap (e.g. CNN output image size,
#                                 e.g. w_resize=320 h_resize=184)
#

uint32 resized_width    # image width, that is, number of columns of image which is output by the cnn
uint32 resized_height   # image height, that is, number of rows of image which is output by the cnn

#
# The CV interface outputs a 8bit bitmap with above <resized> resolution.
# A bit vector is supplied for each pixel in this resized image space.
# The bit vector has the following structure.
#
# most significant bit (large number) -> least significant bit
#        0                           0000                          000
# lane probability                 Lane type                    Lane color
#
# bitmap legend (least to most significant bit ordered):
#
#     Color
#      000    = no lane
#      001    = yellow
#      010    = white
#      011    = orange (Bott's dots can be orange)
# ****************************************
#      Type
#     0000   = no lane
#     0001   = solid
#     0010   = dashed
#     0011   = solid dashed
#     0100   = dashed solid
#     0101   = double solid
#     0110   = double dashed
#     0111   = Bott's dots single
#     1000   = Bott's dots double
# ****************************************
#   lane marking probability
#      0    = no lane
#      1    = lane
uint8[] bitmap
