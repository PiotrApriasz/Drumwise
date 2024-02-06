using System.Security.Claims;
using Drumwise.Application.Common.Interfaces;
using Microsoft.AspNetCore.Http;

namespace Drumwise.Infrastructure.Identity;

public class CurrentUser(IHttpContextAccessor httpContextAccessor) : IUser
{
    public string? Id => httpContextAccessor.HttpContext?.User?.FindFirstValue(ClaimTypes.NameIdentifier);
}