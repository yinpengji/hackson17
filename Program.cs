using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace CaptionImage
{
    static class Program
    {
        // **********************************************
        // *** Update or verify the following values. ***
        // **********************************************

        // Replace the subscriptionKey string value with your valid subscription key.
        const string subscriptionKey = "28e9a946d07547cf8fbf89179921445a";

        // Replace or verify the region.
        //
        // You must use the same region in your REST API call as you used to obtain your subscription keys.
        // For example, if you obtained your subscription keys from the westus region, replace 
        // "westcentralus" in the URI below with "westus".
        //
        // NOTE: Free trial subscription keys are generated in the westcentralus region, so if you are using
        // a free trial subscription key, you should not need to change this region.
        //
        // Also, if you want to use the celebrities model, change "landmarks" to "celebrities" here and in 
        // requestParameters to use the Celebrities model.
        const string uriBase = "https://westus.api.cognitive.microsoft.com/vision/v1.0/analyze";

        static Dictionary<string, string> diction = new Dictionary<string, string>();
        static List<string> fileList = new List<string>();

        public static async Task<string> StartMyTask(string imagepath)
        {
            return MakeAnalysisRequest(imagepath).Result;
                // code to execute once foo is done
        }

        static string htmlprefix = @"<html>
<head ><style>
table, tr, th, td {
    border: 1px solid black;
}<style></head>
<body>
<Table>";
        static string htmlpostfix = @"</Table>
</body>
</html>";



        static void Main()
        {
            // Get the path and filename to process from the user.
            Console.WriteLine("Domain-Specific Model:");
            Console.Write("Enter the path to an images you wish to analzye for Description : ");
            string imageFilePath = Console.ReadLine();

            var files = Directory.GetFiles(imageFilePath);
            string content = "";
            foreach (var file in files)
            {
                fileList.Add(file);
                //diction.Add(file, "");
                // Execute the REST API call.
                var dis = StartMyTask(file);
                diction.Add(file, dis.Result);
                content += "<tr><td><image src=\"" + file.Replace("\\", "\\\\") + "\"></td>";
                content += "<td>" + dis.Result + "</td></tr>";
            }
            string all = htmlprefix + content + htmlpostfix;
            Console.WriteLine(all);
            File.WriteAllText("C:\\videoextract\\stroy.html", all);
            
            Console.WriteLine("\nPlease wait a moment for the results to appear. Then, press Enter to exit ...\n");
            Console.ReadLine();
        }


        /// <summary>
        /// Gets a thumbnail image from the specified image file by using the Computer Vision REST API.
        /// </summary>
        /// <param name="imageFilePath">The image file to use to create the thumbnail image.</param>
        static async Task<string> MakeAnalysisRequest(string imageFilePath)
        {
            HttpClient client = new HttpClient();

            // Request headers.
            client.DefaultRequestHeaders.Add("Ocp-Apim-Subscription-Key", subscriptionKey);

            // Request parameters. Change "landmarks" to "celebrities" here and in uriBase to use the Celebrities model.
            string requestParameters = "visualFeatures=Description&details=Celebrities&language=en";

            // Assemble the URI for the REST API Call.
            string uri = uriBase + "?" + requestParameters;

            HttpResponseMessage response;

            // Request body. Posts a locally stored JPEG image.
            byte[] byteData = GetImageAsByteArray(imageFilePath);
            string description = "";
            using (ByteArrayContent content = new ByteArrayContent(byteData))
            {
                // This example uses content type "application/octet-stream".
                // The other content types you can use are "application/json" and "multipart/form-data".
                content.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");

                // Execute the REST API call.
                response = await client.PostAsync(uri, content);

                // Get the JSON response.
                string contentString = await response.Content.ReadAsStringAsync();

                // Display the JSON response.
                //Console.WriteLine("\nResponse:\n");
                //Console.WriteLine(JsonPrettyPrint(contentString));
                dynamic jObj = JsonConvert.DeserializeObject(contentString);
                if (jObj != null && jObj.description != null && jObj.description.captions != null)
                {
                    Console.WriteLine(imageFilePath);
                    foreach (dynamic obj in jObj.description.captions)
                    {
                        Console.WriteLine(obj.text);
                        description = obj.text;
                    }
                }
            }
            return description;
        }


        /// <summary>
        /// Returns the contents of the specified file as a byte array.
        /// </summary>
        /// <param name="imageFilePath">The image file to read.</param>
        /// <returns>The byte array of the image data.</returns>
        static byte[] GetImageAsByteArray(string imageFilePath)
        {
            FileStream fileStream = new FileStream(imageFilePath, FileMode.Open, FileAccess.Read);
            BinaryReader binaryReader = new BinaryReader(fileStream);
            return binaryReader.ReadBytes((int)fileStream.Length);
        }


        /// <summary>
        /// Formats the given JSON string by adding line breaks and indents.
        /// </summary>
        /// <param name="json">The raw JSON string to format.</param>
        /// <returns>The formatted JSON string.</returns>
        static string JsonPrettyPrint(string json)
        {
            if (string.IsNullOrEmpty(json))
                return string.Empty;

            json = json.Replace(Environment.NewLine, "").Replace("\t", "");

            StringBuilder sb = new StringBuilder();
            bool quote = false;
            bool ignore = false;
            int offset = 0;
            int indentLength = 3;

            foreach (char ch in json)
            {
                switch (ch)
                {
                    case '"':
                        if (!ignore) quote = !quote;
                        break;
                    case '\'':
                        if (quote) ignore = !ignore;
                        break;
                }

                if (quote)
                    sb.Append(ch);
                else
                {
                    switch (ch)
                    {
                        case '{':
                        case '[':
                            sb.Append(ch);
                            sb.Append(Environment.NewLine);
                            sb.Append(new string(' ', ++offset * indentLength));
                            break;
                        case '}':
                        case ']':
                            sb.Append(Environment.NewLine);
                            sb.Append(new string(' ', --offset * indentLength));
                            sb.Append(ch);
                            break;
                        case ',':
                            sb.Append(ch);
                            sb.Append(Environment.NewLine);
                            sb.Append(new string(' ', offset * indentLength));
                            break;
                        case ':':
                            sb.Append(ch);
                            sb.Append(' ');
                            break;
                        default:
                            if (ch != ' ') sb.Append(ch);
                            break;
                    }
                }
            }

            return sb.ToString().Trim();
        }
    }
}
