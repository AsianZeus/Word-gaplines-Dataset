
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:/Users/akroc/Desktop/pirdstest-070ad07a9e78.json"
word_dict={}

def detect_document(path):
    
    wholepage=''
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            # print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                # print('Paragraph confidence: {}'.format(
                #     paragraph.confidence))
                wholepage+='\n'
                i=1
                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    # print('Word text: {} (confidence: {})'.format(
                    #     word_text, word.confidence))
                    wholepage+=f" {word_text}"
                    # word_dict.update({path+str(i) : word_text})
                    
                    # for symbol in word.symbols:
                    #     print('\tSymbol: {} (confidence: {})'.format(
                    #         symbol.text, symbol.confidence))
                    i+=1
    return wholepage
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


from autocorrect import Speller
import cv2 as cv
i=1
wholepage=''
pathx = 'C:/Users/akroc/Desktop/AI and ML/PIRDS/Dataset/CroppedImage'
filename = os.listdir(pathx)
for name in filename:
    if(i>20):
        break
    newpath= pathx+'/'+'Image14.png'
    # orignal =cv.imread(newpath)
    # cv.namedWindow('img', 0)
    # cv.imshow("img",orignal)
    # cv.waitKey(0)
    wholepage=detect_document(newpath)
    print(wholepage)
    spell = Speller(lang='en')
    print(f"\n\nConverted Text: \n\n{spell(wholepage)}")
    i+=1
    break
# print(word_dict)
'''
You already have credentials that are suitable for this purpose.

Don't want to use this existing service account? Create a new service account

pirdstest
Email address	
pirdstest@pirdstest.iam.gserviceaccount.com
Key IDs	
070ad07a9e7895b3cf1f9ae91ac081854844e2b9'''