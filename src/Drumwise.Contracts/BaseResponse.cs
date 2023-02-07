namespace Drumwise.Contracts;

public abstract class BaseResponse
{
    public required bool Success { get; set; }
    public required string Message { get; set; }
}