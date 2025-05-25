import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Ellipsis, UploadCloud } from 'lucide-react';
import { useState } from 'react';

export default function Main() {
  const handleClear = () => {
    const fileInput = document.getElementById('upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const [uploading, setUploading] = useState(false);
  const handleFileUpload = () => {
    const fileInput = document.getElementById('upload') as HTMLInputElement;
    if (fileInput.files && fileInput.files.length > 0) {
      setUploading(true);
      setTimeout(() => {
        setUploading(false);
      }, 2000);
    } else {
      console.log('没有选择文件');
    }
  };

  return (
    <>
      <div className="bg-background text-foreground h-screen w-screen">
        <main className="flex h-full w-full items-center justify-center p-4">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle>DNS 分类</CardTitle>
              <CardDescription>
                基于多尺度卷积神经网络的 DNS 分类系统
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col w-full max-w-sm gap-3">
                <Label htmlFor="upload">选择文件</Label>
                <Input id="upload" type="file" accept=".pcap" />
              </div>

              <footer className="p-4 text-center text-sm text-muted-foreground">
                <p>© 2023 DNS 分类系统</p>
              </footer>
            </CardContent>
            <CardFooter>
              <div className="flex w-full justify-end gap-3">
                <Button variant="outline" onClick={handleClear}>
                  清除
                </Button>
                <Button
                  className="text-white"
                  onClick={handleFileUpload}
                  disabled={uploading}
                >
                  {uploading ? <Ellipsis /> : <UploadCloud />}
                  上传文件
                </Button>
              </div>
            </CardFooter>
          </Card>
        </main>
      </div>
    </>
  );
}
