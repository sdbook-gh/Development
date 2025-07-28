using UnityEngine;
using UnityEngine.UI;
// using UnityEngine.Networking;
// using System.Collections;
using System.IO;
using System;

using static ImageLibrary;

public class HelloWorld : MonoBehaviour
{
    public Text displayText;
    public RawImage rawImageComponent;
    ImageUtils imageUtils = new ImageUtils();

    void Start()
    {
        GameObject gameObject;
        gameObject = GameObject.Find("MyRawImage");
        if (gameObject != null)
        {
            rawImageComponent = gameObject.GetComponent<RawImage>();
            LoadAndShow();
            bool res = imageUtils.loadImage();
            ulong pointer = imageUtils.getImageBufferPointer();
            int size = imageUtils.getImageBufferSize();
            Debug.Log($"ImageUtils loadImage result: {res}, pointer: {pointer}, size: {size}");
            // Span<byte> dataSpan = null;
            // unsafe {
            //     dataSpan = new System.Span<byte>((byte*)pointer, (int)size);
            // }
            // byte[] byteArray = dataSpan.ToArray();
        }
    }

    void LoadAndShow()
    {
        // 1. 读字节
        byte[] bytes;
        bytes = File.ReadAllBytes("e:/dev/SDK/unity_project/My project/image.jpg");
        if (bytes == null || bytes.Length == 0)
        {
            Debug.LogError("文件读取出错");
            return;
        }
        // 2. 创建临时 Texture2D（任意格式，仅用于解码）
        Texture2D temp = new Texture2D(2, 2); // 尺寸无所谓，LoadImage 会重设
        if (!temp.LoadImage(bytes))          // Unity 自动解码 JPG/PNG
        {
            Debug.LogError("LoadImage 失败");
            Destroy(temp);
            return;
        }
        // 3. 转成 RGB24（去掉 alpha，节约内存）
        Texture2D rgb = new Texture2D(temp.width, temp.height, TextureFormat.RGB24, false);
        Color32[] pixels = temp.GetPixels32();   // 取出像素
        rgb.SetPixels32(pixels);
        rgb.Apply();
        // 4. 显示
        if (rawImageComponent != null)
            rawImageComponent.texture = rgb;
        // 5. 清理
        Destroy(temp);
        Debug.Log($"JPG 加载成功: {rgb.width} x {rgb.height}, 格式: {rgb.format}");
    }
}
