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
import * as api from '@/lib/api';
import { redirect } from 'react-router';
import Header from '@/components/Header';

export default function Main() {
  const [message, setMessage] = useState<string | null>(null);

  const handleClear = () => {
    const fileInput = document.getElementById('upload') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = '';
    }
    setMessage(null);
    setUploading(false);
  };

  const [uploading, setUploading] = useState(false);
  const handleFileUpload = async () => {
    const fileInput = document.getElementById('upload') as HTMLInputElement;
    if (fileInput.files && fileInput.files.length > 0) {
      setUploading(true);
      const fileSize = (fileInput.files?.[0].size || 0) / 1024;
      if (fileSize > 10 * 1024 * 1024) {
        setMessage(`文件大小${fileInput.files?.[0].size}超过限制（10MB）`);
        setUploading(false);
        return;
      }
      if (!fileInput.files?.[0].name.endsWith('.pcap')) {
        setMessage('请上传有效的 .pcap 文件');
        setUploading(false);
        return;
      }
      const results = await api.predict(fileInput.files[0]);
      if (results.message) {
        setMessage(results.message);
      } else {
        redirect(`/report/${results.report_id}`);
      }

      setUploading(false);
    } else {
      setMessage('没有选择文件');
    }
  };

  return (
    <>
      <div className="bg-background text-foreground h-screen w-screen">
        <Header />
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
                <Input
                  id="upload"
                  type="file"
                  accept=".pcap"
                  onChange={() => {
                    setMessage(null);
                  }}
                />
                {message && (
                  <div className="text-red-500 text-sm">{message}</div>
                )}
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
